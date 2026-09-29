"""V6 encoder lookup matches sealed closed-chain FK without simulator state."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.mechanism_observer import ClosedChainHeightModel
from wheeled_tasks.chassis.step_trigger import estimate_height_if_grounded


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "contracts/v6_height_lookup_v1.json"


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE_PATH.read_text())


@pytest.fixture(scope="module")
def builder():
    path = ROOT / "scripts/build_v6_height_lookup.py"
    spec = importlib.util.spec_from_file_location("height_lookup_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LookupBuilder(module.DEFAULT_ASSET, module.DEFAULT_MANIFEST_SHA256)


def test_generated_table_is_bound_to_current_sealed_asset_and_generator(table, builder):
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert table["asset_manifest_sha256"] == sha(builder.bundle / "manifest.json")
    assert table["model_spec_sha256"] == sha(builder.bundle / "model_spec.json")
    assert table["solver_sha256"] == sha(builder.solver_path)
    assert table["fk_sha256"] == sha(builder.bundle / "tools/v6_mechanism.py")
    assert table["generator_sha256"] == sha(ROOT / "scripts/build_v6_height_lookup.py")
    assert table["hardware_calibrated"] is False
    assert table["passive_knee_domain_deg"] == [40., 105.]
    assert table["max_closure_gap_m"] < 1e-8
    for side in table["sides"].values():
        assert len(side["delta_rad"]) == 257
        assert min(side["passive_inner_deg"]) == 40.
        assert max(side["passive_inner_deg"]) == 105.
        checks = side["validation"]["checks"]
        assert sum(row["kind"] == "midpoint" for row in checks) == 256
        assert sum(row["kind"] == "common_rotation" for row in checks) == 36
        assert max(row["error_mm"] for row in checks) < .01


def test_runtime_encoder_lookup_matches_independent_full_fk_at_nominal_and_rotation(table, builder):
    observer = ClosedChainHeightModel(table, "cpu")
    names = table["control_joint_names"]
    rows, expected = [], []
    for common in (0., -.7, 1.4):
        q = dict(builder.nominal)
        for angle in np.linspace(0., common, 15)[1:]:
            fixed = {key: value + angle if key in ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")
                     else value for key, value in builder.active.items()}
            q = builder.solve(fixed, q)
        rows.append([q[name] for name in names])
        expected.append([builder.wheel(side, q)[0] for side in ("L", "R")])
    points, valid = observer.wheel_positions(torch.tensor(rows))
    assert valid.all()
    assert np.max(np.linalg.norm(points.numpy() - np.asarray(expected), axis=-1)) < 1e-5


def test_unbounded_common_hip_rotation_does_not_change_relative_geometry(table, builder):
    observer = ClosedChainHeightModel(table, "cpu")
    q = torch.tensor([[builder.nominal[name] for name in table["control_joint_names"]]])
    point, valid = observer.wheel_positions(q)
    q[:, [0, 1, 3, 4]] += 4 * math.pi
    rotated, rotated_valid = observer.wheel_positions(q)
    assert valid.all() and rotated_valid.all()
    torch.testing.assert_close(point, rotated, atol=2e-6, rtol=0.)


def test_outside_table_and_nan_have_per_side_invalidity_no_extrapolation(table, builder):
    observer = ClosedChainHeightModel(table, "cpu")
    q = torch.tensor([[builder.nominal[name] for name in table["control_joint_names"]]]).repeat(3, 1)
    q[0, 1] = q[0, 0] + table["sides"]["left"]["delta_rad"][-1] + .01
    q[1, 4] = q[1, 3] + table["sides"]["right"]["delta_rad"][0] - .01
    q[2, 0] = float("nan")
    points, valid = observer.wheel_positions(q)
    assert valid.tolist() == [[False, True], [True, False], [False, True]]
    assert torch.isfinite(points).all()
    assert (points[~valid] == 0.).all()
    height, support_valid = estimate_height_if_grounded(
        points, torch.tensor([[0., 0., -1.]]).expand(3, -1), observer.wheel_radius_m,
        grounded=torch.ones(3, 2, dtype=torch.bool), sensor_valid=valid,
        wheel_axle_b=observer.wheel_axles_b,
    )
    assert support_valid.tolist() == valid.tolist()
    assert height[valid].min() > .2


@pytest.mark.parametrize("field,value", [
    ("delta_rad", [0., 0.]), ("wheel_center_b_m", [[0., 0.]]),
    ("hip_axis_b", [0., 2., 0.]), ("wheel_radius_m", -1.),
])
def test_invalid_lookup_fails_before_tensor_execution(table, field, value):
    bad = copy.deepcopy(table)
    bad["sides"]["left"][field] = value
    with pytest.raises(ValueError, match="Invalid left"):
        ClosedChainHeightModel(bad, "cpu")


def test_wrong_encoder_coordinate_order_rejected(table):
    bad = copy.deepcopy(table)
    bad["control_joint_names"][0:2] = reversed(bad["control_joint_names"][0:2])
    with pytest.raises(ValueError, match="six V6 model coordinates"):
        ClosedChainHeightModel(bad, "cpu")


def test_generator_rejects_unbound_asset_identity(builder):
    with pytest.raises(ValueError, match="frozen identity"):
        type(builder)(builder.bundle, "0" * 64)
