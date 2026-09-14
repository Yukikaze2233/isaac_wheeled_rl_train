"""CPU FK/closure and USD composition checks against the actual saved chassis."""
import hashlib
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wheeled_tasks.v40.repaired_visuals import ASSET_SHA256, RepairedKinematics, RepairedVisuals

MODEL = REPO.parent / "model/纯底盘"
pytestmark = pytest.mark.skipif(not MODEL.is_dir(), reason="optional portable chassis not installed")


@pytest.fixture
def model():
    return RepairedKinematics(MODEL)


def coordinates(model, left=65., right=72.):
    q = dict(L_joint1=.63, R_joint1=-.23, L_joint3=2.3, R_joint3=-1.7)
    for side, angle, name in (("L", left, "L_joint2"), ("R", right, "R_jonit2")):
        data = model.data["sides"][side]
        q[name] = (np.deg2rad(angle)-np.pi-data["source_knee_origin_z_rad"])/data["source_knee_axis_sign"]
    return q


@pytest.mark.parametrize("angles", [(35., 80.), (65., 72.), (34.9, 80.1)])
def test_matches_actual_serial_urdf_six_link_fk(model, angles):
    """Independent scipy/robot.urdf FK catches root convention and axis-sign errors."""
    q = coordinates(model, *angles)
    quat = Rotation.from_euler("xyz", [.2, -.31, 1.2]).as_quat()
    root = np.eye(4)
    root[:3, :3] = Rotation.from_quat(quat).as_matrix()
    root[:3, 3] = [1., -2., .37]
    fk = {"base_link": root}
    urdf = ET.parse(REPO / "assets/urdf_v40/robot.urdf").getroot()
    for joint in urdf.findall("joint"):
        origin = joint.find("origin")
        matrix = np.eye(4)
        matrix[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
        matrix[:3, :3] = (Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
                           @ Rotation.from_rotvec(np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                                                   * q[joint.get("name")]).as_matrix())
        fk[joint.find("child").get("link")] = fk[joint.find("parent").get("link")] @ matrix
    matrices, diagnostics = model.pose(root[:3, 3], quat, q)
    for side in ("L", "R"):
        for part, link in (("C2", "link1"), ("shank", "link2"), ("wheel", "link3")):
            np.testing.assert_allclose(matrices[f"{side}_{part}"], fk[f"{side}_{link}"], atol=1e-12)
        assert diagnostics[side]["max_pin_gap_m"] < 2e-8
    np.testing.assert_allclose(matrices["base"], root @ model.canonical, atol=1e-12)
    assert diagnostics["L"]["outside_preview_range"] == (angles[0] < 35)


def test_both_closures_move_with_raw_knee_and_preserve_root_wheel(model):
    first, _ = model.pose([0, 0, .32], [0, 0, 0, 1], coordinates(model, 40, 45))
    second, _ = model.pose([0, 0, .32], [0, 0, 0, 1], coordinates(model, 75, 78))
    for side in ("L", "R"):
        for part in ("C0", "C1", "C3", "C4"):
            assert not np.allclose(first[f"{side}_{part}"], second[f"{side}_{part}"])
    q = coordinates(model, 40, 45)
    q["L_joint3"] += .5
    spun, _ = model.pose([0, 0, .32], [0, 0, 0, 1], q)
    np.testing.assert_allclose(spun["L_wheel"][:3, 3], first["L_wheel"][:3, 3])
    assert not np.allclose(spun["L_wheel"][:3, :3], first["L_wheel"][:3, :3])
    quat = Rotation.from_euler("xyz", [.7, .4, -.9]).as_quat()
    moved, _ = model.pose([3, -4, .82], quat, q)
    shift = model.root_matrix([3, -4, .82], quat) @ np.linalg.inv(model.root_matrix([0, 0, .32], [0, 0, 0, 1]))
    for part in moved:
        np.testing.assert_allclose(moved[part], shift @ spun[part], atol=1e-12)


def test_reject_nonfinite_and_unreachable_without_clipping(model):
    q = coordinates(model)
    q["L_joint3"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.pose([0, 0, .32], [0, 0, 0, 1], q)
    with pytest.raises(ValueError, match="nonfinite"):
        model.pose([0, 0, float("inf")], [0, 0, 0, 1], coordinates(model))
    with pytest.raises(ValueError, match="unreachable"):
        model._circle(np.zeros(2), 1., np.array([3., 0.]), 1., 1)
    # A real unachievable knee angle near the measured upper-loop tangency.
    failures = 0
    for angle in np.linspace(0, 180, 1001):
        try:
            model.pose([0, 0, .32], [0, 0, 0, 1], coordinates(model, angle, angle))
        except ValueError as exc:
            assert "unreachable" in str(exc)
            failures += 1
    assert failures > 0


@pytest.mark.parametrize("filename", ["chassis.usdc", "kinematics.json"])
def test_source_hash_rejected(model, tmp_path, filename):
    for name in ("chassis.usdc", "kinematics.json", "manifest.json"):
        shutil.copyfile(MODEL / name, tmp_path / name)
    with (tmp_path / filename).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        RepairedKinematics(tmp_path)


def test_render_only_composition_and_restore_visibility(model, tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    import torch

    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    UsdPhysics.RigidBodyAPI.Apply(robot.GetPrim())
    collision = UsdGeom.Cube.Define(stage, str(robot.GetPath()) + "/Collider")
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    physics_before = stage.GetRootLayer().ExportToString()
    overlay = RepairedVisuals(stage, model, str(robot.GetPath()), tmp_path)
    q = coordinates(model)
    env = SimpleNamespace(common_step_counter=3, joint_names=list(q),
                          robot=SimpleNamespace(data=SimpleNamespace(
                              root_link_pos_w=SimpleNamespace(torch=torch.tensor([[1., 2., .32]])),
                              root_link_quat_w=SimpleNamespace(torch=torch.tensor([[0., 0., 0., 1.]])))),
                          _joint_state=lambda: (torch.tensor([list(q.values())]), None))
    overlay.update(env)
    assert robot.GetPrim().IsActive()
    assert robot.GetVisibilityAttr().Get() == "invisible"
    assert UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get()
    root = stage.GetPrimAtPath(overlay.PATH)
    assert sum(prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(root)) == 15
    assert not any("Physics" in api for prim in Usd.PrimRange(root) for api in prim.GetAppliedSchemas())
    matrices, _ = model.pose([1., 2., float(env.robot.data.root_link_pos_w.torch[0, 2])],
                             [0, 0, 0, 1], dict(zip(q, env._joint_state()[0][0].tolist())))
    cache = UsdGeom.XformCache()
    for name, matrix in matrices.items():
        prim = stage.GetPrimAtPath(overlay.PATH + "/" + model.names[name])
        np.testing.assert_allclose(np.asarray(cache.GetLocalToWorldTransform(prim)).T, matrix, atol=1e-12)
    overlay.close()
    overlay.close()
    assert robot.GetVisibilityAttr().Get() == "inherited"
    assert robot.GetPrim().IsActive() and robot.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert 'physics:collisionEnabled = false' not in stage.GetRootLayer().ExportToString()
    assert 'PhysicsRigidBodyAPI' in physics_before
    assert hashlib.sha256((MODEL / "chassis.usdc").read_bytes()).hexdigest() == ASSET_SHA256
