"""V7 relative knee coordinates, checked against tracked geometry and NumPy FK.

These CPU checks establish the own-asset angle definition. They do not map the
reference robot's rear2 coordinates, change physical stops, or qualify motion.
"""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.leg_pose_mapping import LegPoseMapper
from wheeled_tasks.chassis.relative_knee_geometry import RelativeKneeGeometry


ROOT = Path(__file__).resolve().parents[1]
ASSET = (ROOT / "reports/v7_training_review_20261011/mechanical_stop_candidates_20261011"
         / "stop107_reference_v2/asset")
TABLE = ROOT / "configs/v7_formal_20261012/leg_pose_107.json"
KNEES = ("L_joint2", "R_jonit2")
ACTIVE_LEGS = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")


@pytest.fixture(scope="module")
def asset():
    spec = json.loads((ASSET / "model_spec.json").read_text())
    manifest = json.loads((ASSET / "manifest.json").read_text())
    table = json.loads(TABLE.read_text())
    loader = importlib.util.spec_from_file_location(
        "relative_knee_independent_asset_fk", ASSET / "tools/v6_mechanism.py")
    reference = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(reference)
    return spec, manifest, table, reference


@pytest.fixture(scope="module")
def mapper(asset):
    spec, manifest, table, _ = asset
    return LegPoseMapper(spec, manifest, table, "cpu", dtype=torch.float64)


def nominal_rows(asset, count=1, dtype=torch.float64):
    manifest = asset[1]
    return torch.tensor([manifest["nominal_joint_pos"][name]
                         for name in manifest["tree_joint_names"]], dtype=dtype).repeat(count, 1)


def independent_inner_angles(asset, rows):
    """Check CAD hinge/frame angle using world axes, without zero arithmetic.

    Unprojected 3D hip/knee/wheel LINK-origin vectors have a different datum;
    their axis-normal projection is independently checked below.
    """
    spec, manifest, _, reference = asset
    result = []
    for row in rows:
        frames = reference.fk(spec, dict(zip(manifest["tree_joint_names"], row.tolist())))
        angles = []
        for side in ("L", "R"):
            upper = frames[side + "_link1"][:3, 0]
            lower = frames[side + "_link2"][:3, 0]
            cosine = np.dot(upper, lower) / (np.linalg.norm(upper) * np.linalg.norm(lower))
            # This bounds roundoff in the independent inverse trig, not the
            # physical observation passed to the production geometry helper.
            angles.append(math.acos(float(np.clip(-cosine, -1., 1.))))
        result.append(angles)
    return rows.new_tensor(result)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tracked_branch_matches_mapper_and_independent_world_axis_fk(asset, mapper, dtype):
    spec, manifest, table, _ = asset
    rows = []
    for side in table["sides"]:
        branch = side["full_q"]
        rows.extend(branch[index] for index in (0, len(branch) // 3, len(branch) // 2, -1))
    q = torch.tensor(rows, dtype=dtype)
    before = q.clone()
    geometry = RelativeKneeGeometry(spec, manifest, manifest["tree_joint_names"])
    actual = geometry.inner_angles_rad(q)
    tolerance = 2e-7 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, independent_inner_angles(asset, q), rtol=0., atol=tolerance)
    expected = mapper.metrics(q.double())["knee_inner_deg"].to(dtype) * (math.pi / 180.)
    torch.testing.assert_close(actual, expected, rtol=0., atol=tolerance)
    assert actual.dtype == dtype and actual.device == q.device
    assert actual.shape == (len(rows), 2)
    assert torch.equal(q, before)


def test_nominal_projected_pin_vectors_match_inner_angle_despite_axial_offset(asset):
    spec, manifest, _, reference = asset
    q = nominal_rows(asset)
    geometry = RelativeKneeGeometry(spec, manifest, manifest["tree_joint_names"])
    actual = geometry.inner_angles_rad(q)[0]
    frames = reference.fk(spec, dict(zip(manifest["tree_joint_names"], q[0].tolist())))
    joints = {joint["name"]: joint for joint in spec["joints"]}
    for index, (side, knee_name) in enumerate(zip(("L", "R"), KNEES)):
        hip, knee, wheel = (frames[side + f"_link{number}"][:3, 3] for number in (1, 2, 3))
        axis = frames[side + "_link2"][:3, :3] @ np.asarray(joints[knee_name]["axis"])
        axis /= np.linalg.norm(axis)
        thigh, shank = hip - knee, wheel - knee

        def angle(first, second):
            cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
            return math.acos(float(np.clip(cosine, -1., 1.)))

        unprojected = angle(thigh, shank)
        projected = angle(thigh - axis * np.dot(thigh, axis), shank - axis * np.dot(shank, axis))
        assert abs(np.dot(shank, axis)) == pytest.approx(.015, abs=1e-12)
        assert projected == pytest.approx(actual[index].item(), abs=2e-12)
        assert abs(math.degrees(unprojected - projected)) > 1e-4


def test_native_order_is_resolved_by_names_not_tree_slots(asset):
    spec, manifest, _, _ = asset
    names = manifest["tree_joint_names"]
    q = nominal_rows(asset, 3)
    q[:, names.index(KNEES[0])] = torch.tensor([-.05, .4, .95])
    q[:, names.index(KNEES[1])] = torch.tensor([-.8, -.25, .04])
    permutation = torch.randperm(len(names), generator=torch.Generator().manual_seed(20261014))
    native_names = [names[index] for index in permutation.tolist()]
    geometry = RelativeKneeGeometry(spec, manifest, native_names)
    assert geometry.joint_indices == tuple(native_names.index(name) for name in KNEES)
    expected = RelativeKneeGeometry(spec, manifest, names).inner_angles_rad(q)
    torch.testing.assert_close(geometry.inner_angles_rad(q[:, permutation]), expected, rtol=0., atol=0.)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_left_and_right_same_physical_angle_and_hard_endpoints(asset, dtype):
    spec, manifest, _, _ = asset
    names = manifest["tree_joint_names"]
    joints = {joint["name"]: joint for joint in spec["joints"]}
    left, right = (joints[name]["limit"] for name in KNEES)
    q = nominal_rows(asset, 4, dtype)
    q[0, names.index(KNEES[0])] = float(left["lower"])
    q[0, names.index(KNEES[1])] = float(right["upper"])
    q[1, names.index(KNEES[0])] = float(left["upper"])
    q[1, names.index(KNEES[1])] = float(right["lower"])
    # Mirror the native signs at two interior positions, rather than assigning
    # the same raw encoder angle to both sides.
    q[2:, names.index(KNEES[0])] = q.new_tensor([.1, .8])
    q[2:, names.index(KNEES[1])] = q.new_tensor([-.1, -.8])
    geometry = RelativeKneeGeometry(spec, manifest, names)
    actual = geometry.inner_angles_rad(q)
    tolerance = 2e-7 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual[:, 0], actual[:, 1], rtol=0., atol=tolerance)
    expected = q.new_tensor([math.radians(40.), math.radians(107.)])[:, None].expand(2, 2)
    torch.testing.assert_close(actual[:2], expected, rtol=0., atol=tolerance)
    assert geometry.inner_limits_rad == pytest.approx((math.radians(40.), math.radians(107.)))
    assert isinstance(geometry.inner_limits_rad, tuple) and len(geometry.inner_limits_rad) == 2
    assert all(type(value) is float for value in geometry.inner_limits_rad)
    low, high = geometry.inner_limits_rad
    # A consumer applying lower/upper margins to this asset's physical span
    # obtains these own-geometry soft bounds; this is not a reward selection.
    assert math.degrees(low + .10 * (high - low)) == pytest.approx(46.7)
    assert math.degrees(high - .05 * (high - low)) == pytest.approx(103.65)


def test_continuous_active_axes_may_rotate_many_turns_without_changing_relative_knee(asset):
    spec, manifest, _, _ = asset
    names = manifest["tree_joint_names"]
    geometry = RelativeKneeGeometry(spec, manifest, names)
    q = nominal_rows(asset, 3)
    baseline = geometry.inner_angles_rad(q)
    for name in ACTIVE_LEGS:
        q[:, names.index(name)] += q.new_tensor([2 * math.pi, -2 * math.pi, 8 * math.pi])
    torch.testing.assert_close(geometry.inner_angles_rad(q), baseline, rtol=0., atol=0.)
    torch.testing.assert_close(independent_inner_angles(asset, q), baseline, rtol=0., atol=2e-12)


def test_relative_angles_outside_physical_domain_are_not_wrapped_or_clipped(asset):
    spec, manifest, _, _ = asset
    names = manifest["tree_joint_names"]
    geometry = RelativeKneeGeometry(spec, manifest, names)
    q = nominal_rows(asset, 2)
    baseline = geometry.inner_angles_rad(q)
    q[:, names.index(KNEES[0])] += q.new_tensor([4 * math.pi, -4 * math.pi])
    q[:, names.index(KNEES[1])] -= q.new_tensor([4 * math.pi, -4 * math.pi])
    expected = baseline + q.new_tensor([4 * math.pi, -4 * math.pi])[:, None]
    torch.testing.assert_close(geometry.inner_angles_rad(q), expected, rtol=0., atol=2e-12)


@pytest.mark.parametrize("fault", [
    "model_kind", "coordinate_schema", "control_frame", "inner_definition", "left_parent", "right_child", "knee_type",
    "left_axis_sign", "right_axis_sign", "axis_not_local_z", "left_limit_span",
    "right_limit_span", "shifted_left_zero", "shifted_right_zero", "reversed_limits", "hinge_origin",
    "manifest_span", "passive_in_control", "active_not_continuous", "active_parent",
    "duplicate_tree_name",
])
def test_wrong_asset_roles_coordinates_and_domains_are_rejected(asset, fault):
    original_spec, original_manifest, _, _ = asset
    spec, manifest = deepcopy(original_spec), deepcopy(original_manifest)
    joints = {joint["name"]: joint for joint in spec["joints"]}
    if fault == "model_kind": manifest["model_kind"] = "another_mechanism"
    elif fault == "coordinate_schema": manifest["coordinate_schema"] = "another_frame"
    elif fault == "control_frame": manifest["control_frame"] = "Xright_Yforward_Zup"
    elif fault == "inner_definition": manifest["knee_angle_definition"] = "motor_shaft_angle"
    elif fault == "left_parent": joints[KNEES[0]]["parent"] = "base_link"
    elif fault == "right_child": joints[KNEES[1]]["child"] = "RR_link2"
    elif fault == "knee_type": joints[KNEES[0]]["type"] = "continuous"
    elif fault == "left_axis_sign": joints[KNEES[0]]["axis"] = [0., 0., -1.]
    elif fault == "right_axis_sign": joints[KNEES[1]]["axis"] = [0., 0., 1.]
    elif fault == "axis_not_local_z": joints[KNEES[0]]["axis"] = [0., 1., 0.]
    elif fault == "left_limit_span": joints[KNEES[0]]["limit"]["upper"] = "1.2"
    elif fault == "right_limit_span": joints[KNEES[1]]["limit"]["lower"] = "-1.2"
    elif fault == "hinge_origin": joints[KNEES[0]]["origin"][0][0] += .1
    elif fault in ("shifted_left_zero", "shifted_right_zero"):
        limits = joints[KNEES[0] if fault == "shifted_left_zero" else KNEES[1]]["limit"]
        for bound in ("lower", "upper"):
            limits[bound] = str(float(limits[bound]) + .05)
    elif fault == "reversed_limits":
        limits = joints[KNEES[0]]["limit"]
        limits["lower"], limits["upper"] = limits["upper"], limits["lower"]
    elif fault == "manifest_span": manifest["knee_inner_limits_deg"] = [40., 108.]
    elif fault == "passive_in_control": manifest["control_joint_names"][0] = KNEES[0]
    elif fault == "active_not_continuous": joints[ACTIVE_LEGS[0]]["type"] = "revolute"
    elif fault == "active_parent": joints[ACTIVE_LEGS[0]]["parent"] = "L_link2"
    elif fault == "duplicate_tree_name": manifest["tree_joint_names"][-1] = manifest["tree_joint_names"][0]
    with pytest.raises((ValueError, TypeError)):
        RelativeKneeGeometry(spec, manifest, original_manifest["tree_joint_names"])


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown", "extra", "nonstring"])
def test_native_joint_identity_must_be_complete_unique_and_known(asset, fault):
    spec, manifest, _, _ = asset
    names = list(manifest["tree_joint_names"])
    if fault == "missing": names.remove(KNEES[0])
    elif fault == "duplicate": names[-1] = names[0]
    elif fault == "unknown": names[0] = "unmapped_motor"
    elif fault == "extra": names.append("unmapped_motor")
    elif fault == "nonstring": names[0] = 3
    with pytest.raises((ValueError, TypeError)):
        RelativeKneeGeometry(spec, manifest, names)


@pytest.mark.parametrize("fault", ["one_dimensional", "missing_column", "extra_dimension", "integer", "float16"])
def test_observation_requires_complete_floating_native_joint_rows(asset, fault):
    spec, manifest, _, _ = asset
    geometry = RelativeKneeGeometry(spec, manifest, manifest["tree_joint_names"])
    q = nominal_rows(asset, 2)
    if fault == "one_dimensional": q = q[0]
    elif fault == "missing_column": q = q[:, :-1]
    elif fault == "extra_dimension": q = q[:, None, :]
    elif fault == "integer": q = q.long()
    elif fault == "float16": q = q.half()
    with pytest.raises((ValueError, TypeError)):
        geometry.inner_angles_rad(q)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonfinite_knee_values_remain_for_caller_classification_without_input_repair(asset, dtype):
    spec, manifest, _, _ = asset
    names = manifest["tree_joint_names"]
    geometry = RelativeKneeGeometry(spec, manifest, names)
    q = nominal_rows(asset, 4, dtype)
    expected_finite = geometry.inner_angles_rad(q)
    q[0, names.index(KNEES[0])] = torch.nan
    q[1, names.index(KNEES[1])] = torch.inf
    q[2, names.index(KNEES[0])] = -torch.inf
    # A nonfinite unrelated motor does not fabricate a knee measurement. The
    # caller must still handle that lane's whole physical safety state.
    q[3, names.index(ACTIVE_LEGS[0])] = torch.nan
    before = q.clone()
    actual = geometry.inner_angles_rad(q)
    assert torch.isnan(actual[0, 0])
    assert torch.isneginf(actual[1, 1]) and torch.isneginf(actual[2, 0])
    torch.testing.assert_close(actual[0, 1], expected_finite[0, 1], rtol=0., atol=0.)
    torch.testing.assert_close(actual[1, 0], expected_finite[1, 0], rtol=0., atol=0.)
    torch.testing.assert_close(actual[2, 1], expected_finite[2, 1], rtol=0., atol=0.)
    torch.testing.assert_close(actual[3], expected_finite[3], rtol=0., atol=0.)
    torch.testing.assert_close(q, before, rtol=0., atol=0., equal_nan=True)
