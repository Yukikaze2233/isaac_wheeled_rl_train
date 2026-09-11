"""Offline V40 asset contract tests. Never import Isaac or step a simulator.

Run with PYTHONDONTWRITEBYTECODE=1 and pytest -p no:cacheprovider. MuJoCo tests
are skipped if unavailable; persisted manifest must still disclose a blocked
candidate. The exact source snapshot is inside this repository, not Downloads.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets/urdf_v40"
GENERATOR = REPO / "tools/prepare_v40_assets.py"
SPEC = importlib.util.spec_from_file_location("v40_asset_generator", GENERATOR)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


@pytest.fixture(scope="module")
def source():
    return ET.parse(ASSETS / "source.urdf").getroot()


@pytest.fixture(scope="module")
def canonical():
    return ET.parse(ASSETS / "robot.urdf").getroot()


@pytest.fixture(scope="module")
def manifest():
    return json.loads((ASSETS / "manifest.json").read_text())


def test_import_safe_without_simulator_and_no_processes_or_writes(tmp_path):
    code = r"""
import builtins, importlib.util, os, pathlib, subprocess, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('mujoco', 'isaaclab', 'isaacsim', 'omni', 'torch'):
        raise AssertionError('simulator/heavy SDK imported at module import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def forbid(*args, **kwargs): raise AssertionError('side effect during import')
subprocess.Popen = os.system = forbid
pathlib.Path.write_text = pathlib.Path.write_bytes = pathlib.Path.mkdir = forbid
spec = importlib.util.spec_from_file_location('generator_import_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.build) and callable(module.main)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(GENERATOR)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_manifest_contract_and_closed_training_gate(manifest):
    required = {"schema_version", "robot_id", "control_frame", "total_mass_kg", "knee_inner_limits_deg", "urdf", "mjcf", "source_sha256", "files_sha256", "collision_validation", "hardware_deployment_ready", "nominal_joint_pos", "nominal_base_height_m"}
    required |= {"base_visual_bounds_m", "isaac_cooking_tested", "server_adjacency_filter_verified", "adjacent_collision_filter_pairs"}
    assert set(manifest) == required
    assert manifest["isaac_cooking_tested"] is False
    assert manifest["server_adjacency_filter_verified"] is False
    assert manifest["schema_version"] == 1
    assert manifest["robot_id"] == "own_v40"
    assert manifest["control_frame"] == "Xforward_Yleft_Zup"
    assert manifest["urdf"] == "robot.urdf" and manifest["mjcf"] == "inspection.xml"
    assert manifest["hardware_deployment_ready"] is False
    assert manifest["collision_validation"]["passed"] is False  # Both hips are pending material/assembly and CAD body-assignment review.
    assert manifest["collision_validation"]["details"]["candidate_status"] == "blocked_candidate"
    assert manifest["collision_validation"]["details"]["blockers"]
    assert manifest["nominal_base_height_m"] == .32
    assert manifest["nominal_joint_pos"] == assets.NOMINAL_JOINT_POS
    assert manifest["knee_inner_limits_deg"] == [35, 80]
    assert manifest["collision_validation"]["details"]["physics_steps_executed"] == 0


def test_complete_hash_manifest_and_portable_references(manifest, canonical):
    # The raw inventory remains immutable; the independently approved research
    # layer owns exactly these two additional records, never retrofits the raw SHA.
    variant_records = {assets.RESEARCH_APPROVAL_FILE, assets.RESEARCH_MANIFEST_FILE}
    expected = {p.relative_to(ASSETS).as_posix() for p in ASSETS.rglob("*") if p.is_file() and p.relative_to(ASSETS).as_posix() not in variant_records | {"manifest.json"}}
    assert set(manifest["files_sha256"]) == expected
    for relative, digest in manifest["files_sha256"].items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert assets.digest(ASSETS / relative) == digest
    assert assets.digest(ASSETS / "source.urdf") == manifest["source_sha256"] == assets.SOURCE_SHA256
    for mesh in canonical.iter("mesh"):
        assert assets.portable_mesh_path(ASSETS / "robot.urdf", mesh.get("filename")).is_file()
    for xml_name in ("inspection.xml", "inspection_whole_hulls.xml"):
        xml = ET.parse(ASSETS / xml_name).getroot()
        for mesh in xml.findall("asset/mesh"):
            assert assets.portable_mesh_path(ASSETS / xml_name, mesh.get("file")).is_file()
    with pytest.raises(ValueError):
        assets.portable_mesh_path(ASSETS / "robot.urdf", "../private.STL")


def test_source_visual_bytes_and_inertial_coefficients_unchanged(source, canonical):
    provenance = json.loads((ASSETS / "provenance.json").read_text())
    old, _, base = assets.topology(source)
    new, _, _ = assets.topology(canonical)
    for name in old:
        assert len(new[name].findall("visual")) == 1
        assert old[name].find("visual/geometry/mesh").attrib == new[name].find("visual/geometry/mesh").attrib
        assert old[name].find("inertial/mass").attrib == new[name].find("inertial/mass").attrib
        assert old[name].find("inertial/inertia").attrib == new[name].find("inertial/inertia").attrib
        uri = old[name].find("visual/geometry/mesh").get("filename")
        assert assets.digest(ASSETS / uri) == provenance["source_visual_meshes_sha256"][uri]
        if name != base:
            for kind in ("visual", "inertial"):
                assert old[name].find(f"{kind}/origin").attrib == new[name].find(f"{kind}/origin").attrib
        # Every partition keeps the original origin (root is rotated exactly once).
        expected = assets.canonicalize(source).find(f"link[@name='{name}']/collision/origin").attrib
        if name not in assets.WHEEL_LINKS:
            assert all(c.find("origin").attrib == expected for c in new[name].findall("collision"))
        else:
            assert new[name].find("collision/geometry/cylinder") is not None


def test_only_root_origins_are_reexpressed(source):
    result = assets.canonicalize(source)
    old_links, old_joints, base = assets.topology(source)
    new_links, new_joints, _ = assets.topology(result)
    for name in old_links:
        for kind in ("inertial", "visual", "collision"):
            original, changed = old_links[name].find(kind), new_links[name].find(kind)
            if name == base:
                expected = np.eye(4)
                expected[:3, :3] = assets.CONTROL_ROTATION
                np.testing.assert_allclose(assets.transform(changed), expected @ assets.transform(original), atol=1e-14)
            else:
                assert ET.tostring(original) == ET.tostring(changed)
    for name in old_joints:
        assert old_joints[name].find("axis").attrib == new_joints[name].find("axis").attrib
        assert old_joints[name].find("limit").attrib == new_joints[name].find("limit").attrib
        if old_joints[name].find("parent").get("link") != base:
            assert ET.tostring(old_joints[name]) == ET.tostring(new_joints[name])
    np.testing.assert_array_equal(assets.CONTROL_ROTATION @ [0, -1, 0], [1, 0, 0])
    np.testing.assert_array_equal(assets.CONTROL_ROTATION @ [1, 0, 0], [0, 1, 0])


def test_sampled_fk_com_inertia_covariance(source):
    report = assets.validate_canonical(source, assets.canonicalize(source))
    assert report["passed"] and report["samples"] >= 32
    assert report["max_transform_error"] < 1e-12
    assert report["max_com_error_m"] < 1e-12
    assert report["max_inertia_error_kg_m2"] < 1e-12
    np.testing.assert_allclose(report["root_com_control_m"], [-.005, .0003, -.053], atol=1e-14)
    np.testing.assert_allclose(report["root_inertia_control_kg_m2"], [[.178, -.000264, -.004123], [-.000264, .126, -.00049], [-.004123, -.00049, .255]], atol=1e-14)
    assert report["total_mass_kg"] == pytest.approx(12.752, abs=1e-12)


def test_hard_knee_limits_continuous_joints_and_nominal(canonical):
    _, joints, _ = assets.topology(canonical)
    assert set(joints) == set(assets.ACTION_ORDER)
    assert "R_jonit2" in joints and "R_joint2" not in joints
    for name, joint in joints.items():
        if name not in assets.KNEE_LIMITS:
            assert joint.get("type") == "continuous"
        else:
            assert joint.get("type") == "revolute"
            lo, hi = [float(joint.find("limit").get(k)) for k in ("lower", "upper")]
            np.testing.assert_allclose([lo, hi], assets.KNEE_LIMITS[name], atol=1e-12, rtol=0)
            assert lo <= assets.NOMINAL_JOINT_POS[name] <= hi
            rz = assets.vec(joint.find("origin").get("rpy"))[2]
            axis_z = assets.vec(joint.find("axis").get("xyz"))[2]
            beta = sorted(math.degrees(math.pi + rz + axis_z * q) for q in (lo, hi))
            np.testing.assert_allclose(beta, [35, 80], atol=1e-10)
    # Explicitly reject previously suggested 36/40 cm candidates, not "fix" them.
    assert .670019132664772 > assets.KNEE_LIMITS["L_joint2"][1]
    assert -.634824495282681 < assets.KNEE_LIMITS["R_jonit2"][0]


def test_component_coverage_and_conservative_convexity(canonical):
    scipy = pytest.importorskip("scipy.spatial")
    report = json.loads((ASSETS / "collision_report.json").read_text())
    expected_counts = {"base_link": 14, "L_link1": 5, "L_link2": 5, "L_link3": 1, "R_link1": 5, "R_link2": 5, "R_link3": 1}
    for name, expected in expected_counts.items():
        entry = report["links"][name]
        triangles = assets.stl_triangles(ASSETS / entry["source_mesh"])
        vertices, faces, labels, count = assets.connected_mesh(triangles)
        assert count == expected == entry["component_count"]
        assert len(canonical.findall(f"link[@name='{name}']/collision")) == count
        assert sum(p["source_triangles"] for p in entry["pieces"]) == len(triangles)
        assert sum(p["source_vertices"] for p in entry["pieces"]) == len(vertices)
        for piece in entry["pieces"]:
            if piece["shape"] == "cylinder":
                radial = np.linalg.norm(vertices[:, :2], axis=1)
                assert radial.max() <= piece["radius_m"] + 1e-12
                assert np.abs(vertices[:, 2] - piece["center_link_m"][2]).max() <= .5 * piece["length_m"] + 1e-12
                assert entry["legacy_mesh_envelopes"][0]["hull_vertex_count"] == 720
                continue
            points = np.array([[float(x) for x in line.split()[1:]] for line in (ASSETS / piece["mesh"]).read_text().splitlines() if line.startswith("v ")])
            hull = scipy.ConvexHull(points)
            original = vertices[labels == piece["component_id"]]
            for start in range(0, len(original), 2048):
                assert (original[start:start + 2048] @ hull.equations[:, :3].T + hull.equations[:, 3]).max() <= 1e-9
            np.testing.assert_allclose(points.min(0), original.min(0), atol=1e-12, rtol=0)
            np.testing.assert_allclose(points.max(0), original.max(0), atol=1e-12, rtol=0)
    assert report["geometry_removed"] is False and report["geometry_shrunk"] is False
    assert report["mass_or_inertia_reestimated"] is False
    assert report["certified_concave_decomposition"] is False


def test_nominal_geometry_tangency_and_no_height_workaround(canonical):
    clearances = assets.geometry_clearances(canonical, ASSETS / "source.urdf")
    for name in ("L_link3", "R_link3"):
        assert abs(clearances[name]["visual_mesh_min_z_m"]) < 1e-7
    for name in set(clearances) - {"L_link3", "R_link3"}:
        assert clearances[name]["visual_mesh_min_z_m"] > 0


def test_mjcf_collision_and_diagnostic_actuator_contract():
    root = ET.parse(ASSETS / "inspection.xml").getroot()
    assert root.find("compiler").get("inertiafromgeom") == "false"
    assert [(e.get("body1"), e.get("body2")) for e in root.findall("contact/exclude")] == [(p["body1"], p["body2"]) for p in assets.adjacent_pairs(ET.parse(ASSETS / "robot.urdf").getroot())]
    assert root.find("option/flag").get("filterparent") == "disable"
    assert not root.findall("contact/pair")
    for geom in root.iter("geom"):
        if "_visual_" in geom.get("name"):
            assert geom.get("contype") == geom.get("conaffinity") == "0"
        else:
            assert geom.get("contype") == geom.get("conaffinity") == "1"
    motors = root.findall("actuator/motor")
    assert tuple(m.get("joint") for m in motors) == assets.ACTION_ORDER
    assert all(m.get("ctrlrange") == "-1 1" for m in motors)
    assert root.find("keyframe/key").get("name") == "nominal_candidate"


def test_mujoco_static_parity_without_stepping(canonical, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    def forbidden(*args, **kwargs):
        raise AssertionError("Static verification must not step physics")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)
    report = assets.static_inspection(canonical, ASSETS / "inspection.xml")
    assert report["passed_frame_mass_inertia"]
    assert report["physics_steps_executed"] == report["simulation_time_s"] == 0
    assert report["nq"] == 13 and report["nv"] == 12 and report["nu"] == 6
    assert not any(report["warnings"])
    assert set(report["joint_addresses_resolved_by_name"]) == set(assets.ACTION_ORDER)
    assert len(report["custom_contact_exclusions"]) == 6
    assert report["explicit_six_pair_filter_verified_in_mujoco"]
    assert report["implicit_parent_filter_disabled"]
    assert report["minimum_nonadjacent_geom_distance"]["distance_m"] > 0.006


def test_source_hash_pin_and_existing_output_protection(source, tmp_path):
    fake = tmp_path / "source.urdf"
    fake.write_text(ET.tostring(source, encoding="unicode"))
    with pytest.raises(ValueError, match="byte-exact"):
        assets.validate_source(fake, source)
    with pytest.raises(ValueError, match="already exists"):
        assets.build(ASSETS / "source.urdf", ASSETS)


def test_rebuild_independent_from_parent_repo_without_mujoco(tmp_path):
    pytest.importorskip("scipy")
    output = tmp_path / "rebuilt"
    result = assets.build(ASSETS / "source.urdf", output, run_mujoco=False)
    assert result["collision_validation"]["passed"] is False
    assert not (output / assets.RESEARCH_APPROVAL_FILE).exists()
    assert not (output / assets.RESEARCH_MANIFEST_FILE).exists()
    assert result["collision_validation"]["details"]["mujoco_available"] is False
    assert any("skipped" in s for s in result["collision_validation"]["details"]["blockers"])
    for relative in ("robot.urdf", "inspection.xml", "source.urdf", "inspection_whole_hulls.xml"):
        assert assets.digest(output / relative) == assets.digest(ASSETS / relative)
    # Diagnostic round-off can vary with NumPy/BLAS; geometry bytes may not.
    rebuilt_frame = json.loads((output / "frame_validation.json").read_text())
    saved_frame = json.loads((ASSETS / "frame_validation.json").read_text())
    for key in ("max_axis_error", "max_com_error_m", "max_inertia_error_kg_m2", "max_transform_error"):
        assert 0 <= rebuilt_frame.pop(key) < 1e-12
        assert 0 <= saved_frame.pop(key) < 1e-12
    # Python 3.12 changed float summation; diagnostic mass is not byte-stable.
    np.testing.assert_allclose(rebuilt_frame.pop("total_mass_kg"), saved_frame.pop("total_mass_kg"),
                               atol=1e-12, rtol=0)
    assert rebuilt_frame == saved_frame
    for original in (ASSETS / "collisions").iterdir():
        assert assets.digest(output / "collisions" / original.name) == assets.digest(original)
    # No optional evidence paths or upstream reports are needed for this hold.
    assert (output / assets.HIP_MATERIAL_REVIEW_HOLD_PATH).read_text() == assets.HIP_MATERIAL_REVIEW_HOLD_JSON
    assert result["collision_validation"]["details"]["material_assembly_review_hold"]["pending_joints"] == ["L_joint1", "R_joint1"]
    assert [p["joint"] for p in result["adjacent_collision_filter_pairs"] if not p["geometry_review_supported"]] == ["L_joint1", "R_joint1"]
    frozen = json.loads((ASSETS / "evidence/hip_review_status_sync_baseline.json").read_text())
    for relative, digest in frozen["physical_files_sha256"].items():
        assert assets.digest(output / relative) == digest  # All 49 physical files, including both baseline MJCFs/SRDF.


def test_mjcf_keyframe_follows_named_emission_not_action_index(canonical):
    shuffled = ET.fromstring(ET.tostring(canonical))
    joints = shuffled.findall("joint")
    for joint in joints:
        shuffled.remove(joint)
    for joint in reversed(joints):
        shuffled.append(joint)
    xml = assets.make_mjcf(shuffled)
    emitted = [joint.get("name") for joint in xml.find("worldbody").iter("joint")]
    assert emitted != list(assets.ACTION_ORDER)
    qpos = assets.vec(xml.find("keyframe/key").get("qpos"))
    np.testing.assert_allclose(qpos[7:], [assets.NOMINAL_JOINT_POS[name] for name in emitted], atol=1e-15)
    assert [m.get("joint") for m in xml.findall("actuator/motor")] == list(assets.ACTION_ORDER)


def test_collision_audit_reports_improvement_without_false_readiness(manifest):
    report = json.loads((ASSETS / "static_validation.json").read_text())
    if not report["components"]["available"] or not report["whole_link_baseline"]["available"]:
        pytest.skip("Persisted asset was built without MuJoCo; blocked manifest is checked separately")
    def penetrations(contacts):
        return [c for c in contacts if c["body1"] != "world" and c["body2"] != "world" and c["distance_m"] < -1e-6]
    assert len(penetrations(report["whole_link_baseline"]["contacts_default_parent_filter"])) == 2
    assert not penetrations(report["legacy_component_mesh"]["contacts_default_parent_filter"])
    assert not penetrations(report["components"]["contacts_active_policy"])
    assert len(penetrations(report["legacy_component_mesh"]["contacts_including_joint_adjacent_bodies"])) == 73
    all_pair = penetrations(report["components"]["contacts_including_joint_adjacent_bodies"])
    assert len(all_pair) == manifest["collision_validation"]["details"]["all_pair_penetrating_self_contact_count"] > 0
    assert manifest["collision_validation"]["passed"] is False
    screen = manifest["collision_validation"]["details"]["physx_portability_screen"]
    assert screen["isaac_physx_cooking_verified"] is False
    assert screen["over_budget_parts"] == []
    assert screen["active_wheel_collision_type"] == "cylinder"


def test_measured_cylinder_geometry_and_mjcf_half_length(canonical):
    report = json.loads((ASSETS / "collision_report.json").read_text())
    mjcf = ET.parse(ASSETS / "inspection.xml").getroot()
    for name in assets.WHEEL_LINKS:
        collision = canonical.find(f"link[@name='{name}']/collision")
        primitive = collision.find("geometry/cylinder")
        assert primitive is not None and collision.find("geometry/mesh") is None
        measured = report["wheel_cylinders"][name]
        vertices = assets.stl_triangles(ASSETS / f"meshes/{name}.STL").reshape(-1, 3)
        assert float(primitive.get("radius")) == np.linalg.norm(vertices[:, :2], axis=1).max()
        assert float(primitive.get("length")) == np.ptp(vertices[:, 2])
        np.testing.assert_allclose(assets.origin(collision)[0], [0, 0, .5 * (vertices[:, 2].min() + vertices[:, 2].max())], atol=1e-15)
        geom = next(g for g in mjcf.iter("geom") if g.get("name") == f"{name}_collision_000")
        assert geom.get("type") == "cylinder" and geom.get("mesh") is None
        np.testing.assert_allclose(assets.vec(geom.get("size")), [measured["radius_m"], .5 * measured["length_m"]], atol=1e-15)
        assert measured["max_source_radial_outside_m"] < 1e-12
        assert measured["max_source_axial_outside_m"] < 1e-12
        assert not measured["collision_geometry_shrunk"]
    for path in (ASSETS / "collisions").iterdir():
        assert not path.name.startswith(("L_link3", "R_link3"))


def test_preserved_73_contact_evidence_is_byte_exact():
    old_manifest = json.loads((ASSETS / "evidence/revision1_manifest.json").read_text())
    old_static = ASSETS / "evidence/revision1_static_validation.json"
    assert assets.digest(old_static) == old_manifest["files_sha256"]["static_validation.json"]
    assert assets.digest(ASSETS / "evidence/revision1_collision_report.json") == old_manifest["files_sha256"]["collision_report.json"]
    contacts = json.loads(old_static.read_text())["components"]["contacts_including_joint_adjacent_bodies"]
    assert len([c for c in contacts if c["body1"] != "world" and c["body2"] != "world" and c["distance_m"] < -1e-6]) == 73


def test_adjacency_review_does_not_hide_real_source_material(manifest):
    report = json.loads((ASSETS / "adjacency_review.json").read_text())
    assert report["source_winding_is_watertight_certificate"] is False
    assert report["review_supported"] is False
    assert len(report["pairs"]) == 6
    assert sum(p["witness_count"] for p in report["pairs"]) == 880
    assert [p["joint"] for p in report["pairs"] if not p["review_supported"]] == ["L_joint1", "R_joint1"]
    right = next(p for p in report["pairs"] if p["joint"] == "R_joint1")
    assert right["unexplained_witness_count"] == 0  # Preserve the old finite sample, NOT its former policy inference.
    assert right["limited_sample_no_unexplained_witnesses"] is True
    assert right["material_assembly_review_status"] == "pending"
    witnesses = manifest["collision_validation"]["details"]["unexplained_material_overlap_witnesses"]
    assert len(witnesses) == 3
    for witness in witnesses:
        assert witness["joint"] == "L_joint1"
        assert abs(witness["parent_winding"] - 1) < 1e-6
        assert abs(witness["child_winding"] - 1) < 1e-6
        assert .025 < witness["radial_distance_about_joint_axis_m"] < .027
        for key in ("parent_surface_check", "child_surface_check"):
            assert witness[key]["three_ray_inside_parities"] == [1, 1, 1]
            assert witness[key]["nearest_source_triangle_distance_m"] > .0008
    filters = manifest["adjacent_collision_filter_pairs"]
    assert len(filters) == 6
    assert [p["joint"] for p in filters if p["policy_status"] == "proposed_pending_material_review"] == ["L_joint1", "R_joint1"]
    assert manifest["collision_validation"]["passed"] is False


def test_exact_named_filter_contract_in_srdf(canonical, manifest):
    expected = [(p["body1"], p["body2"]) for p in assets.adjacent_pairs(canonical)]
    srdf = ET.parse(ASSETS / "collision_filters.srdf").getroot()
    assert [(e.get("link1"), e.get("link2")) for e in srdf.findall("disable_collisions")] == expected
    assert len(set(tuple(sorted(p)) for p in expected)) == 6
    assert ("base_link", "L_link2") not in expected and ("base_link", "R_link2") not in expected
    assert manifest["server_adjacency_filter_verified"] is False


def test_canonical_base_visual_bounds_for_gpu_clearance(canonical, manifest):
    bounds = np.array(manifest["base_visual_bounds_m"])
    np.testing.assert_allclose(bounds, assets.base_visual_bounds(canonical, ASSETS / "source.urdf"), atol=1e-15)
    assert bounds.shape == (2, 3) and np.all(bounds[0] < bounds[1])
    np.testing.assert_allclose(bounds[:, 2], [-.17299328744411469, .07804601639509201], atol=1e-15)
    assert bounds[0, 0] < -.283 and bounds[1, 0] > .283


def test_bilateral_hip_comparison_snapshot_is_source_bound_and_portable(manifest):
    path = ASSETS / assets.HIP_MATERIAL_REVIEW_HOLD_PATH
    summary = json.loads(path.read_text())
    hold = manifest["collision_validation"]["details"]["material_assembly_review_hold"]
    assert path.read_text() == assets.HIP_MATERIAL_REVIEW_HOLD_JSON
    assert assets.digest(path) == hold["evidence_sha256"]
    assert summary["pending_joints"] == hold["pending_joints"] == ["L_joint1", "R_joint1"]
    assert summary["user_decision"] == "repair_CAD_and_body_assignment_first_no_V4_training"
    assert summary["training_permitted"] is False
    assert summary["automatic_hip_filter_approval_permitted"] is False
    for relative, digest in summary["source_hashes"].items():
        assert assets.digest(ASSETS / relative) == digest
    for relative, digest in summary["upstream_reports_sha256"].items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert len(digest) == 64
    point = summary["right_actual_nominal_overlap"]
    assert point["id"] == "Rm-10"
    assert point["q_R"] == assets.NOMINAL_JOINT_POS["R_joint1"]
    assert point["q_L"] == assets.NOMINAL_JOINT_POS["L_joint1"]
    assert point["base"]["surface_distance_mm"] == pytest.approx(.06871886856810629)
    assert point["thigh"]["surface_distance_mm"] == pytest.approx(.0735362748459427)
    for key in ("base", "thigh"):
        assert point[key]["classification"] == "inside_evidence"
        assert point[key]["three_ray_inside_parities"] == [1, 1, 1]
        assert abs(point[key]["winding"] - 1) < 1e-8
    assert summary["old_sampling"]["right_hip_jointly_occupied_witnesses"] == 0
    assert [x["verified_both_inside_count"] for x in summary["targeted_source_section_scan"]["rows"]] == [12, 12, 12, 12]


def test_review_state_sync_preserves_all_physical_and_historical_bytes():
    baseline = json.loads((ASSETS / "evidence/hip_review_status_sync_baseline.json").read_text())
    assert len(baseline["physical_files_sha256"]) == 49
    for relative, digest in {**baseline["physical_files_sha256"], **baseline["retained_diagnostics_sha256"]}.items():
        assert assets.digest(ASSETS / relative) == digest
    archive = ASSETS / "evidence/revision2_adjacency_review.json"
    assert assets.digest(archive) == baseline["prior_adjacency_review_sha256"]
    before, after = json.loads(archive.read_text()), json.loads((ASSETS / "adjacency_review.json").read_text())
    prior = {p["joint"]: p for p in before["pairs"]}
    for current in after["pairs"]:
        old = prior[current["joint"]]
        for key in ("convex_pair_intersections", "witness_count", "unexplained_witness_count", "full_intersection_radial_max_m", "bounds_in_child_joint_frame_m"):
            assert current[key] == old[key]
    assert next(p for p in before["pairs"] if p["joint"] == "R_joint1")["review_supported"] is True  # Historical inference only.
    assert next(p for p in after["pairs"] if p["joint"] == "R_joint1")["review_supported"] is False


def test_clean_finite_samples_cannot_clear_bilateral_cad_hold():
    report = {"review_supported": True, "blockers": [], "pairs": [
        {"joint": name, "review_supported": True, "witness_count": 40, "unexplained_witness_count": 0,
         "interpretation": "Synthetic clean limited sample, not an approval"} for name in assets.ACTION_ORDER]}
    assets.apply_hip_material_review_hold(report)
    assert report["review_supported"] is False
    assert [p["joint"] for p in report["pairs"] if not p["review_supported"]] == ["L_joint1", "R_joint1"]
    assert len(report["blockers"]) == 2
    first = json.dumps(report, sort_keys=True)
    assets.apply_hip_material_review_hold(report)
    assert json.dumps(report, sort_keys=True) == first  # Idempotent even when build also applies the hold.


@pytest.fixture
def research_snapshot(tmp_path, manifest):
    """A byte-identical raw snapshot; negative fixtures never modify real assets."""
    directory = tmp_path / "immutable-raw"
    directory.mkdir()
    for relative in ["manifest.json", *manifest["files_sha256"]]:
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ASSETS / relative, target)
    approval = tmp_path / "explicit-decision.json"
    shutil.copyfile(ASSETS / assets.RESEARCH_APPROVAL_FILE, approval)
    return directory, approval


def test_research_manifest_is_a_separate_hash_bound_layer(manifest):
    approval = json.loads((ASSETS / assets.RESEARCH_APPROVAL_FILE).read_text())
    variant = json.loads((ASSETS / assets.RESEARCH_MANIFEST_FILE).read_text())
    assert approval["approval_source"] == {"kind": "explicit_user_decision", "question_id": "v40-equivalent-research-scope", "selected_option": "同意，先用串联等效研究模型训练"}
    assert approval["raw_manifest_sha256"] == assets.digest(ASSETS / "manifest.json") == "ec93ff2bab06b817796338268d9640754849c49796cec88138b1d24b5a6ce196"
    assert manifest["collision_validation"]["passed"] is False
    assert variant["collision_validation"]["passed"] is True
    assert variant["collision_validation"]["scope"] == assets.RESEARCH_COLLISION_SCOPE
    model = variant["research_model"]
    assert model["scope_id"] == assets.RESEARCH_SCOPE_ID
    assert model["raw_manifest_file"] == "manifest.json" and model["approval_file"] == assets.RESEARCH_APPROVAL_FILE
    assert model["raw_manifest_sha256"] == approval["raw_manifest_sha256"]
    assert model["approval_sha256"] == assets.digest(ASSETS / assets.RESEARCH_APPROVAL_FILE)
    assert model["hardware_deployment_approved"] is False and variant["hardware_deployment_ready"] is False
    assert model["raw_material_review_stays_unresolved"] is True
    assert variant["raw_material_review"]["hip_geometry_review_supported"] == {"L_joint1": False, "R_joint1": False}
    assert variant["raw_material_review"]["source_material_overlap_repaired"] is False
    assert model["dynamics_prior"]["model"] == "equivalent_serial_open_chain_research"
    for key in ("robot_id", "control_frame", "urdf", "mjcf", "source_sha256", "total_mass_kg", "knee_inner_limits_deg", "nominal_joint_pos", "nominal_base_height_m", "base_visual_bounds_m"):
        assert variant[key] == manifest[key]
    assert variant["files_sha256"] == {**manifest["files_sha256"], "manifest.json": approval["raw_manifest_sha256"], assets.RESEARCH_APPROVAL_FILE: model["approval_sha256"]}
    assert assets.RESEARCH_MANIFEST_FILE not in variant["files_sha256"]
    assert assets.RESEARCH_APPROVAL_FILE not in manifest["files_sha256"]
    for relative, digest in variant["files_sha256"].items():
        assert assets.digest(ASSETS / relative) == digest
    assert set(variant["files_sha256"]) == {p.relative_to(ASSETS).as_posix() for p in ASSETS.rglob("*") if p.is_file() and p.name != assets.RESEARCH_MANIFEST_FILE}
    for raw_pair, approved, active in zip(manifest["adjacent_collision_filter_pairs"], approval["adjacent_collision_filter_pairs"], variant["adjacent_collision_filter_pairs"], strict=True):
        assert approved == active
        assert {k: active[k] for k in ("body1", "body2", "joint", "geometry_review_supported")} == {k: raw_pair[k] for k in ("body1", "body2", "joint", "geometry_review_supported")}
        assert active["raw_policy_status"] == raw_pair["policy_status"]
        assert active["research_exclusion_approved"] is True
        assert active["policy_status"] == "user_approved_joint_internal_contact_exclusion"
    assert all(variant["collision_validation"]["details"]["checks"].values())
    assert variant["collision_validation"]["details"]["nonadjacent_minimum_clearance_m"] > .006
    assert variant["source_geometry_modified"] is False and variant["global_self_collision_enabled"] is True
    assert variant["nonadjacent_collisions_unchanged"] is True
    assert variant["collision_validation"]["details"]["new_simulation_run_for_variant"] is False


def test_research_requires_explicit_approval_and_cli_pair(tmp_path):
    with pytest.raises(ValueError, match="explicit research approval"):
        assets.make_research_manifest(ASSETS, None)
    before = assets.digest(ASSETS / "manifest.json")
    result = subprocess.run([sys.executable, "-B", str(GENERATOR), "--raw-assets", str(ASSETS), "--out", str(ASSETS)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and "must be supplied together" in result.stderr
    assert assets.digest(ASSETS / "manifest.json") == before


@pytest.mark.parametrize("key,value", [
    ("approved", False), ("scope_id", "another-scope"), ("raw_manifest_sha256", "0" * 64),
    ("hardware_deployment_approved", True), ("source_geometry_modified", True),
    ("global_self_collision_enabled", False), ("nonadjacent_collisions_unchanged", False),
    ("raw_material_review_stays_unresolved", False), ("knee_inner_limits_deg", [0, 180]),
    ("approval_source", {"kind": "inferred_from_clean_sampling"}),
])
def test_research_rejects_missing_or_broadened_approval(tmp_path, key, value):
    approval = json.loads((ASSETS / assets.RESEARCH_APPROVAL_FILE).read_text())
    approval[key] = value
    path = tmp_path / "unapproved.json"
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError):
        assets.make_research_manifest(ASSETS, path)


@pytest.mark.parametrize("change", ["missing", "nonadjacent", "erase_material_failure", "deny_exclusion", "rewrite_raw_policy"])
def test_research_rejects_altered_joint_pair_scope(tmp_path, change):
    approval = json.loads((ASSETS / assets.RESEARCH_APPROVAL_FILE).read_text())
    pairs = approval["adjacent_collision_filter_pairs"]
    if change == "missing": pairs.pop()
    elif change == "nonadjacent": pairs[0]["body2"] = "R_link2"
    elif change == "erase_material_failure": pairs[0]["geometry_review_supported"] = True
    elif change == "deny_exclusion": pairs[0]["research_exclusion_approved"] = False
    else: pairs[0]["raw_policy_status"] = "source_material_repaired"
    path = tmp_path / "bad-pairs.json"
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="exactly the six"):
        assets.make_research_manifest(ASSETS, path)


def test_research_rejects_raw_hash_drift(research_snapshot):
    directory, approval = research_snapshot
    with (directory / "collision_report.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        assets.emit_research_variant(directory, approval)
    assert not (directory / assets.RESEARCH_MANIFEST_FILE).exists()
    assert not (directory / assets.RESEARCH_APPROVAL_FILE).exists()


@pytest.mark.parametrize("failure", ["nonadjacent", "wheel_ground"])
def test_valid_approval_does_not_waive_failed_static_checks(research_snapshot, failure):
    # Synthetic, explicitly re-bound negative fixtures only; production never
    # retargets an approval or changes a report to make its static gate pass.
    directory, approval_path = research_snapshot
    relative = "static_validation.json" if failure == "nonadjacent" else "collision_report.json"
    data = json.loads((directory / relative).read_text())
    if failure == "nonadjacent": data["components"]["minimum_nonadjacent_geom_distance"]["distance_m"] = -.001
    else: data["wheel_cylinders"]["R_link3"]["nominal_collision_min_z_m"] = -.001
    assets.write_json(directory / relative, data)
    raw = json.loads((directory / "manifest.json").read_text())
    raw["files_sha256"][relative] = assets.digest(directory / relative)
    assets.write_json(directory / "manifest.json", raw)
    approval = json.loads(approval_path.read_text())
    approval["raw_manifest_sha256"] = assets.digest(directory / "manifest.json")
    assets.write_json(approval_path, approval)
    variant = assets.make_research_manifest(directory, approval_path)
    assert variant["collision_validation"]["passed"] is False
    assert variant["collision_validation"]["details"]["blockers"]
    assert variant["research_model"]["research_exclusion_approved"] is True


def test_research_rebuild_is_byte_reproducible_idempotent_and_no_simulator(research_snapshot, monkeypatch):
    import builtins
    directory, approval_path = research_snapshot
    raw = json.loads((directory / "manifest.json").read_text())
    before = {relative: assets.digest(directory / relative) for relative in ["manifest.json", *raw["files_sha256"]]}
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("mujoco", "isaaclab", "isaacsim", "omni"):
            raise AssertionError("Record-only research emission must not import a simulator")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = assets.emit_research_variant(directory, approval_path)
    assert result["collision_validation"]["passed"] is True
    assert (directory / assets.RESEARCH_MANIFEST_FILE).read_bytes() == (ASSETS / assets.RESEARCH_MANIFEST_FILE).read_bytes()
    assert (directory / assets.RESEARCH_APPROVAL_FILE).read_bytes() == (ASSETS / assets.RESEARCH_APPROVAL_FILE).read_bytes()
    again = assets.emit_research_variant(directory, directory / assets.RESEARCH_APPROVAL_FILE)
    assert result == again
    for relative, digest in before.items():
        assert assets.digest(directory / relative) == digest
    assert {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()} == set(before) | {assets.RESEARCH_APPROVAL_FILE, assets.RESEARCH_MANIFEST_FILE}
    (directory / assets.RESEARCH_MANIFEST_FILE).write_text("{}")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        assets.emit_research_variant(directory, approval_path)
    assert (directory / assets.RESEARCH_MANIFEST_FILE).read_text() == "{}"


def test_wheel_ground_micro_overlap_is_explicit_not_hidden(manifest):
    report = json.loads((ASSETS / "collision_report.json").read_text())
    ground = report["ground_validation"]
    assert ground["passed"] and ground["wheel_proxy_penetration_tolerance_m"] == 1e-4
    assert ground["nonwheel_penetration_tolerance_m"] == 1e-6
    for name in assets.WHEEL_LINKS:
        low = report["wheel_cylinders"][name]["nominal_collision_min_z_m"]
        assert -1e-4 < low < 0
    assert manifest["nominal_base_height_m"] == .32
    assert manifest["nominal_joint_pos"] == assets.NOMINAL_JOINT_POS
