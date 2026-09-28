"""Simulator-free checks of the independently frozen V6 asset contract."""
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/纯底盘_v6_232mm"
BUNDLE = MODEL / "urdf"
MODULE = importlib.util.spec_from_file_location("v6_geometry", MODEL / "tools/v6_mechanism.py")
GEOMETRY = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(GEOMETRY)


def read(name):
    return json.loads((BUNDLE / name).read_text())


def test_all_published_files_match_manifest_and_reviewed_source():
    manifest = read("manifest.json")
    for name, digest in manifest["files_sha256"].items():
        path = (BUNDLE / name).resolve()
        assert path.is_relative_to(BUNDLE.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    assert manifest["source_urdf_sha256"] == "2c061fc49e1cf375b32c19002495aa03c2856941b326688f36cd81d7102e095d"
    assert hashlib.sha256((MODEL / "source/urdf/urdf_v5.0.urdf").read_bytes()).hexdigest() == manifest["source_urdf_sha256"]
    assert manifest["control_frame"] == "Xforward_Yleft_Zup"
    assert manifest["wheel_gear_ratio"] == 15.8
    assert not manifest["hardware_deployment_ready"]


def test_mass_and_inertia_have_explicit_source_and_repair_evidence():
    spec, sources = read("model_spec.json"), read("inertial_sources.json")
    robot = ET.parse(BUNDLE / "source_urdf_v5.0.urdf").getroot()
    source_mass = {link.get("name"): float(link.find("inertial/mass").get("value"))
                   for link in robot.findall("link")}
    assert sum(source_mass.values()) == pytest.approx(13.886)
    assert source_mass["RR_link2"] == source_mass["LL_link2"] == .084
    for body in spec["bodies"]:
        assert body["mass"] == source_mass[body["name"]]
        eig = np.linalg.eigvalsh(body["inertia"])
        assert eig[0] > 0 and eig[0] + eig[1] >= eig[2] - 1e-12
        if sources[body["name"]]["method"] != "source_CAD_preserved":
            assert sources[body["name"]]["owned_mesh_watertight"]
            assert sources[body["name"]]["repair_reason"]
            assert not sources[body["name"]]["hardware_identified"]
    assert sum(item["source_invalid"] for item in sources.values()) == 4
    assert sum(item["method"] != "source_CAD_preserved" for item in sources.values()) == 10


@pytest.mark.parametrize("scale", [0., .4, 1.])
def test_fk_frame_mapping_preserves_physical_geometry_and_power(scale):
    spec, old, mapping = read("model_spec.json"), read("v5_232mm_reference_spec.json"), read("coordinate_mapping.json")
    assert np.array_equal(mapping["source_root_to_control"], np.eye(4))
    local_rotation = GEOMETRY.transform()
    local_rotation[:3, :3] = mapping["v5_local_to_v6_local_rotation"]
    old_q = {name: value * scale for name, value in old["nominal_joint_pos"].items()}
    signs = mapping["joint_sign_from_v5_232mm"]
    new_q = {name: signs[name] * value for name, value in old_q.items()}
    new_frames, old_frames = GEOMETRY.fk(spec, new_q), GEOMETRY.fk(old, old_q)
    for body, old_body in zip(spec["bodies"], old["bodies"]):
        np.testing.assert_allclose(new_frames[body["name"]] @ body["mesh_origin"] @ local_rotation,
                                   old_frames[body["name"]] @ old_body["mesh_origin"], atol=1e-10, rtol=0.)
    sign_vector = np.array([signs[name] for name in spec["nominal_joint_pos"]])
    velocity = np.linspace(-2., 3., len(sign_vector))
    torque = np.linspace(4., -3., len(sign_vector))
    assert (sign_vector * velocity) @ (sign_vector * torque) == pytest.approx(velocity @ torque)
    assert [signs[name] for name in ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1")] == [-1] * 4
    assert signs["L_joint3"] == signs["R_joint3"] == 1


def test_closedchain_and_spring_coordinates_are_physical():
    spec = read("model_spec.json")
    assert len(spec["bodies"]) == 19 and len(spec["joints"]) == 18 and len(spec["constraints"]) == 6
    assert sum(j["type"] == "prismatic" for j in spec["joints"]) == 2
    for pose_name in ("source_closed_joint_pos", "nominal_joint_pos"):
        q = spec[pose_name]
        frames = GEOMETRY.fk(spec, q)
        for closure in spec["constraints"]:
            np.testing.assert_allclose(GEOMETRY.point(frames[closure["body0"]], closure["local_pos0_m"]),
                                       GEOMETRY.point(frames[closure["body1"]], closure["local_pos1_m"]),
                                       atol=2e-7, rtol=0.)
        for name, binding in spec["spring_binding"].items():
            side = name[0]
            joint = next(j for j in spec["joints"] if j["name"] == name)
            assert joint["parent"] == side * 3 + "_link2" and joint["child"] == side * 3 + "_link1"
            assert float(joint["limit"]["upper"]) - float(joint["limit"]["lower"]) == pytest.approx(.080)
            assert binding["full_extension_pin_distance_m"] == pytest.approx(.232)
            assert binding["mesh_long_axis"] == "local_Y" and binding["effort_sign"] == 1
            distance = np.linalg.norm(frames[joint["child"]][:3, 3] - frames[joint["parent"]][:3, 3])
            assert distance == pytest.approx(binding["source_mount_distance_m"] + q[name], abs=1e-10)
            shifted = dict(q, **{name: q[name] + 1e-5})
            other = GEOMETRY.fk(spec, shifted)
            increased = np.linalg.norm(other[joint["child"]][:3, 3] - other[joint["parent"]][:3, 3])
            assert increased - distance == pytest.approx(1e-5, abs=1e-12)


def test_export_has_six_motor_inputs_and_two_separate_internal_spring_inputs():
    xml = ET.parse(BUNDLE / "robot.xml").getroot()
    manifest = read("manifest.json")
    assert ET.parse(BUNDLE / "robot.urdf").getroot().get("name") == manifest["model_kind"]
    assert ET.parse(BUNDLE / "collision_filters.srdf").getroot().get("name") == manifest["model_kind"]
    actuators = xml.find("actuator").findall("motor")
    assert [a.get("joint") for a in actuators] == manifest["control_joint_names"] + manifest["spring_joint_names"]
    assert all(a.get("gear") == "1" for a in actuators)
    for name in manifest["tree_joint_names"]:
        assert float(xml.find(f".//joint[@name='{name}']").get("damping")) == (
            0. if name in manifest["control_joint_names"] else .002)
    assert len(xml.find("equality").findall("connect")) == 6
    assert xml.find("compiler").get("inertiafromgeom") == "false"
    assert xml.find("compiler").get("balanceinertia") != "true"
    assert read("fit_10mpa.json")["max_model_compression_m"] == pytest.approx(.072)
