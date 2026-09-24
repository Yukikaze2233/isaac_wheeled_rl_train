"""Independent V5 exported topology, conservative spring sign and closure regression."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model/纯底盘_v5/urdf"


def read_json(name):
    return json.loads((BUNDLE / name).read_text())


def urdf_fk(robot, coordinates):
    frames = {"base_link": np.eye(4)}
    remaining = list(robot.findall("joint"))
    while remaining:
        count = len(remaining)
        for joint in remaining[:]:
            parent = joint.find("parent").get("link")
            if parent not in frames:
                continue
            node = joint.find("origin")
            t = np.eye(4)
            t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(node.get("rpy"), sep=" ")).as_matrix()
            t[:3, 3] = np.fromstring(node.get("xyz"), sep=" ")
            motion = np.eye(4)
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            value = coordinates[joint.get("name")]
            if joint.get("type") == "prismatic":
                motion[:3, 3] = axis * value
            else:
                motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            frames[joint.find("child").get("link")] = frames[parent] @ t @ motion
            remaining.remove(joint)
        assert len(remaining) < count
    return frames


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(BUNDLE / "robot.xml"))


def test_tree_and_spring_connection_topology(model):
    robot = ET.parse(BUNDLE / "robot.urdf").getroot()
    joints = {j.get("name"): j for j in robot.findall("joint")}
    assert len(robot.findall("link")) == 19 and len(joints) == 18
    assert sum(j.get("type") == "prismatic" for j in joints.values()) == 2
    assert sum(j.get("type") in ("continuous", "revolute") for j in joints.values()) == 16
    assert model.nbody == 20 and model.nv == 24 and model.neq == 6
    constraints = read_json("constraints.json")["constraints"]
    for side in ("L", "R"):
        slider = joints[side + "_spring_slide"]
        assert slider.find("parent").get("link") == side * 3 + "_link2"
        assert slider.find("child").get("link") == side * 3 + "_link1"
        mount = next(c for c in constraints if c["name"] == side + "_spring_upper_mount")
        assert mount["body0"] == side + "_link1" and mount["body1"] == side * 3 + "_link1"
    assert read_json("manifest.json")["active_motor_mapping_verified"] is False


@pytest.mark.parametrize("pose", ["nominal_joint_pos", "source_closed_joint_pos"])
def test_compiled_fk_and_constraint_rank(model, pose):
    spec = read_json("model_spec.json")
    q = spec[pose]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:3] = 0.
    for name, value in q.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    frames = urdf_fk(ET.parse(BUNDLE / "robot.urdf").getroot(), q)
    for name, frame in frames.items():
        np.testing.assert_allclose(data.body(name).xpos, frame[:3, 3], atol=1e-10)
        np.testing.assert_allclose(data.body(name).xmat.reshape(3, 3), frame[:3, :3], atol=1e-10)
    jacobian = data.efc_J.reshape(data.nefc, model.nv)[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    assert np.linalg.matrix_rank(jacobian, tol=1e-8) == 12
    for c in spec["constraints"]:
        np.testing.assert_allclose(data.site(c["name"] + "_0").xpos, data.site(c["name"] + "_1").xpos, atol=1e-7)


def test_engine_restores_passive_slider_perturbation(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = 1.2
    data.qpos[model.joint("L_spring_slide").qposadr[0]] += .001
    mujoco.mj_forward(model, data)
    a, b = model.site("L_spring_upper_mount_0").id, model.site("L_spring_upper_mount_1").id
    initial = np.linalg.norm(data.site_xpos[a] - data.site_xpos[b])
    assert initial > .0009
    for _ in range(200):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.site_xpos[a] - data.site_xpos[b]) < initial / 100
    assert not any(w.number for w in data.warning)


def test_slider_positive_force_extends_and_uses_installed_reference(model):
    bindings = read_json("gas_spring_binding.json")
    spec = read_json("model_spec.json")
    for name, binding in bindings.items():
        assert binding["full_extension_pin_distance_m"] == pytest.approx(.232, abs=1e-9)
        assert binding["stroke_m"] == pytest.approx(.08, abs=1e-9)
        compression = binding["compression_at_q_zero_m"] - spec["nominal_joint_pos"][name]
        assert .05 < compression < .065
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        actuator = model.actuator(name + "_gas_force")
        data.ctrl[actuator.id] = 350.
        mujoco.mj_forward(model, data)
        dof = model.joint(name).dofadr[0]
        assert data.qfrc_actuator[dof] == pytest.approx(350.)
        assert actuator.gear[0] == 1


def test_installed_pin_distances_at_compiled_slider_stops(model):
    usd_text = (BUNDLE / "robot.usda").read_text()
    urdf = ET.parse(BUNDLE / "robot.urdf").getroot()
    for side in ("L", "R"):
        name = side + "_spring_slide"
        binding = read_json("gas_spring_binding.json")[name]
        assert binding["adapter_effective_length_m"] == pytest.approx(.027)
        usd = usd_text.split(f'def PhysicsPrismaticJoint "{name}"', 1)[1].split("}", 1)[0]
        usd_limits = [float(re.search(r"physics:" + key + r"Limit = ([-+\deE.]+)", usd).group(1))
                      for key in ("lower", "upper")]
        limits = urdf.find(f"joint[@name='{name}']/limit")
        compiled = model.joint(name)
        np.testing.assert_allclose(compiled.range, [float(limits.get("lower")), float(limits.get("upper"))], atol=1e-10)
        np.testing.assert_allclose(compiled.range, usd_limits, atol=1e-8)
        for index, expected_ab in enumerate((.152, .232)):
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, 0)
            # Inspect the open spring branch at each stop, independently of leg closure.
            data.qpos[compiled.qposadr[0]] = compiled.range[index]
            mujoco.mj_forward(model, data)
            a, b = data.body(side * 3 + "_link2").xpos, data.body(side * 3 + "_link1").xpos
            assert np.linalg.norm(a - b) == pytest.approx(expected_ab, abs=1e-9)
            compression = binding["compression_at_q_zero_m"] - compiled.range[index]
            assert compression == pytest.approx(.232 - expected_ab, abs=1e-9)


def test_thigh_shank_inner_limits_are_not_motor_angles(model):
    assert read_json("manifest.json")["knee_inner_limits_deg"] == [40., 110.]
    zero_inner = np.pi - 2.3573
    for name, sign in (("L_joint2", 1.), ("R_jonit2", -1.)):
        inner = sorted(np.degrees(zero_inner + sign * model.joint(name).range))
        assert inner == pytest.approx([40., 110.])


def test_mass_uncertainty_is_preserved_and_inertias_are_valid(model):
    assert model.body_mass.sum() == pytest.approx(14.642)
    assert model.body("RR_link2").mass[0] == pytest.approx(.84)
    assert model.body("LL_link2").mass[0] == pytest.approx(.084)
    provenance = read_json("inertial_sources.json")
    assert sum(v["method"] == "owned_mesh_uniform_density_research_prior" for v in provenance.values()) == 10
    for body in model.body_inertia[1:]:
        eig = sorted(body)
        assert eig[0] > 0 and eig[0] + eig[1] >= eig[2] - 1e-10


def test_duplicate_removal_preserves_every_retained_triangle_record():
    ownership = read_json("mesh_ownership.json")
    removed_components = 0
    for name, report in ownership.items():
        source = (ROOT / "model/纯底盘_v5/source/meshes" / (name + ".STL")).read_bytes()
        target = (BUNDLE / "meshes" / (name + ".stl")).read_bytes()
        count = struct.unpack_from("<I", source, 80)[0]
        removed = {i for c in report["removed_duplicate_components"] for i in c["source_face_indices"]}
        expected = b"".join(source[84 + 50 * i:84 + 50 * (i + 1)] for i in range(count) if i not in removed)
        assert target[84:] == expected
        removed_components += len(report["removed_duplicate_components"])
        assert all(c["reference_body"] == "RR_link4" for c in report["removed_duplicate_components"])
    assert removed_components == 14


def test_delivery_hashes_and_engine_evidence_match():
    manifest = read_json("manifest.json")
    actual = {str(p.relative_to(BUNDLE)) for p in BUNDLE.rglob("*") if p.is_file()}
    assert actual == set(manifest["files_sha256"]) | {"manifest.json"}
    for name, expected in manifest["files_sha256"].items():
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == expected
    physx = read_json("physx_validation.json")
    assert physx["status"] == "completed" and physx["simulated_seconds"] >= 12
    assert physx["max_loop_gap_m"] < .001
    for name, expected in physx["validated_model_inputs"].items():
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == expected
    for low, high in physx["sample_ranges"]["compression_m"]:
        assert 0 < low < high < .072
        assert high - low > .005


@pytest.mark.parametrize("name", ["v5_foundation_v1", "v5_mixed_v1", "v5_locomotion_v2"])
def test_base_training_contract_pins_current_asset(name):
    contract = json.loads((ROOT / "contracts" / (name + ".json")).read_text())
    assert contract["asset_manifest_sha256"] == hashlib.sha256((BUNDLE / "manifest.json").read_bytes()).hexdigest()
