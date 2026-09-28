#!/usr/bin/env python3
"""CPU V6 provenance, frame, closure-domain, virtual-work and USD checks.

This validates geometric and export contracts, not hardware safety, self collision,
contact fidelity, policy readiness or a PhysX simulation. No GPU runtime is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics

from build_v6_closedchain import ACTIVE, closure_error, solve_pose, validate
from v6_mechanism import fk, point, transform


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inner_angle(spec, q, side):
    """Pin-center directions projected perpendicular to the common hinge axis.

    Pin origins have different axial representatives. The unprojected 3D angle
    includes that lateral offset and is not the planar thigh/shank inner angle.
    """
    frames = fk(spec, q)
    joint = next(j for j in spec["joints"] if j["child"] == side + "_link2")
    axis = frames[side + "_link2"][:3, :3] @ joint["axis"]
    hip, knee, wheel = (frames[f"{side}_link{i}"][:3, 3] for i in (1, 2, 3))
    thigh, shank = hip - knee, wheel - knee
    thigh -= axis * (axis @ thigh)
    shank -= axis * (axis @ shank)
    return float(np.arccos(np.clip(thigh @ shank / np.linalg.norm(thigh)
                                  / np.linalg.norm(shank), -1., 1.)))


def usd_checks(bundle, spec):
    stage = Usd.Stage.Open(str(bundle / "robot.usda"))
    assert stage and stage.GetDefaultPrim().GetPath() == "/Robot"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    expected = fk(spec, spec["nominal_joint_pos"], transform([0, 0, .32]))
    body_frames = {}
    pose_error = inertia_error = axis_error = pin_error = 0.
    for body in spec["bodies"]:
        prim = stage.GetPrimAtPath("/Robot/" + body["name"])
        matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.)).T
        body_frames[body["name"]] = matrix
        pose_error = max(pose_error, float(np.abs(matrix - expected[body["name"]]).max()))
        mass = UsdPhysics.MassAPI(prim)
        assert abs(mass.GetMassAttr().Get() - body["mass"]) < 1e-6
        quaternion = mass.GetPrincipalAxesAttr().Get()
        rotation = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
        tensor = rotation @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ rotation.T
        inertia_error = max(inertia_error, float(np.abs(tensor - body["inertia"]).max()))
    for joint in spec["joints"]:
        prim = stage.GetPrimAtPath("/Robot/tree_joints/" + joint["name"])
        prism = joint["type"] == "prismatic"
        typed = (UsdPhysics.PrismaticJoint if prism else UsdPhysics.RevoluteJoint)(prim)
        assert prim.IsA(UsdPhysics.PrismaticJoint if prism else UsdPhysics.RevoluteJoint)
        assert not typed.GetExcludeFromArticulationAttr().Get()
        assert not prim.HasAPI(UsdPhysics.DriveAPI, "angular")
        vectors = []
        for index, body in enumerate((joint["parent"], joint["child"])):
            quaternion = prim.GetAttribute(f"physics:localRot{index}").Get()
            rotation = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
            vectors.append(body_frames[body][:3, :3] @ rotation @
                           np.array([1., 0, 0] if prism else [0., 0, 1.]))
        axis_error = max(axis_error, float(np.linalg.norm(np.cross(*vectors))))
        if prism:
            assert abs(typed.GetLowerLimitAttr().Get() - float(joint["limit"]["lower"])) < 1e-7
            assert abs(typed.GetUpperLimitAttr().Get() - float(joint["limit"]["upper"])) < 1e-7
        elif joint["type"] == "revolute":
            for attr, bound in ((typed.GetLowerLimitAttr(), "lower"),
                                (typed.GetUpperLimitAttr(), "upper")):
                if abs(attr.Get() - np.degrees(float(joint["limit"][bound]))) > 1e-5:
                    raise ValueError(f"USD angular limit mismatch: {joint['name']} {bound}")
    for closure in spec["constraints"]:
        joint = UsdPhysics.SphericalJoint(stage.GetPrimAtPath("/Robot/loop_joints/" + closure["name"]))
        assert joint.GetExcludeFromArticulationAttr().Get()
        p0 = point(body_frames[closure["body0"]], joint.GetLocalPos0Attr().Get())
        p1 = point(body_frames[closure["body1"]], joint.GetLocalPos1Attr().Get())
        pin_error = max(pin_error, float(np.linalg.norm(p0 - p1)))
    assert pose_error < 1e-11 and inertia_error < 1e-7 and axis_error < 1e-6 and pin_error < 1e-7
    return {"nominal_body_matrix_max_error": pose_error, "inertia_tensor_max_error_kg_m2": inertia_error,
            "tree_axis_cross_error": axis_error, "loop_pin_max_error_m": pin_error,
            "physx_simulation_run": False, "units": "m_kg_Z_up", "angular_drives": 0}


def run(bundle):
    spec, manifest = load(bundle / "model_spec.json"), load(bundle / "manifest.json")
    mapping, fit = load(bundle / "coordinate_mapping.json"), load(bundle / "fit_10mpa.json")
    for name, sha in manifest["files_sha256"].items():
        assert digest(bundle / name) == sha, f"Bundle hash mismatch: {name}"
    assert digest(bundle / "source_urdf_v5.0.urdf") == manifest["source_urdf_sha256"]
    assert np.array_equal(mapping["source_root_to_control"], np.eye(4))
    assert not mapping["root_extra_Rz90_applied"]
    assert abs(sum(b["mass"] for b in spec["bodies"]) - 13.886) < 1e-12
    assert next(b for b in spec["bodies"] if b["name"] == "RR_link2")["mass"] == .084
    for body in spec["bodies"]:
        eig = np.linalg.eigvalsh(body["inertia"])
        assert eig[0] > 0 and eig[0] + eig[1] >= eig[2] - 1e-12
    topology = validate(bundle, spec)
    old = load(bundle / "v5_232mm_reference_spec.json")
    local_rotation = transform()
    local_rotation[:3, :3] = mapping["v5_local_to_v6_local_rotation"]
    model = mujoco.MjModel.from_xml_path(str(bundle / "robot.xml"))
    data = mujoco.MjData(model)
    urdf = ET.parse(bundle / "robot.urdf").getroot()
    limit_checks = {}
    for joint in spec["joints"]:
        if joint["type"] not in ("revolute", "prismatic"):
            continue
        expected = np.array([float(joint["limit"][key]) for key in ("lower", "upper")])
        compiled = model.joint(joint["name"])
        exported = urdf.find(f"joint[@name='{joint['name']}']/limit")
        actual = np.array([float(exported.get(key)) for key in ("lower", "upper")])
        if (not bool(model.jnt_limited[compiled.id]) or
                not np.allclose(model.jnt_range[compiled.id], expected, rtol=0., atol=1e-12) or
                not np.allclose(actual, expected, rtol=0., atol=1e-12)):
            raise ValueError(f"URDF/MuJoCo limit mismatch: {joint['name']}")
        limit_checks[joint["name"]] = expected.tolist()
    signs = mapping["joint_sign_from_v5_232mm"]
    zero = {side: inner_angle(spec, {}, side) for side in ("L", "R")}
    knee_joints = {"L": "L_joint2", "R": "R_jonit2"}
    for side in ("L", "R"):
        joint = next(j for j in spec["joints"] if j["name"] == knee_joints[side])
        for limit in ("lower", "upper"):
            beta = np.rad2deg(inner_angle(spec, {joint["name"]: float(joint["limit"][limit])}, side))
            assert min(abs(beta - edge) for edge in manifest["knee_inner_limits_deg"]) < 1e-8
    max_closure = max_frame_error = max_angle_error = max_virtual_work_error = 0.
    min_rank = 24
    compressions, q_samples = [], []
    spring_antiderivative = np.polynomial.polynomial.polyint(fit["monomial_coefficients_n"])
    for hip in [-.8, -.42, 0., .42, .8]:
        q = spec["nominal_joint_pos"]
        for beta_deg in np.linspace(*manifest["knee_inner_limits_deg"], 15):
            beta = np.deg2rad(beta_deg)
            prescribed = {"L_joint1": hip, "R_joint1": -hip, "L_joint2": beta - zero["L"],
                          "R_jonit2": -(beta - zero["R"]), "L_joint3": 0., "R_joint3": 0.}
            q = solve_pose(spec, prescribed, q)
            max_closure = max(max_closure, float(np.abs(closure_error(spec, q)).max()))
            for side in ("L", "R"):
                max_angle_error = max(max_angle_error, abs(inner_angle(spec, q, side) - beta))
            new_frames = fk(spec, q)
            old_q = {name: signs[name] * value for name, value in q.items()}
            old_frames = fk(old, old_q)
            for body, old_body in zip(spec["bodies"], old["bodies"]):
                assert body["name"] == old_body["name"]
                new_visual = new_frames[body["name"]] @ body["mesh_origin"] @ local_rotation
                old_visual = old_frames[body["name"]] @ old_body["mesh_origin"]
                max_frame_error = max(max_frame_error, float(np.abs(new_visual - old_visual).max()))
            for name, binding in spec["spring_binding"].items():
                compression = binding["compression_at_q_zero_m"] - q[name]
                assert -1e-8 <= compression <= .080 + 1e-8
                compressions.append(compression)
                side = name[0]
                distance = np.linalg.norm(new_frames[side * 3 + "_link1"][:3, 3] -
                                          new_frames[side * 3 + "_link2"][:3, 3])
                assert abs(distance - (.232 - compression)) < 1e-10
                eps = 1e-6
                energies = [fit["stroke_m"] * np.polynomial.polynomial.polyval(
                    (compression + offset) / fit["stroke_m"], spring_antiderivative)
                    for offset in (-eps, eps)]
                force = np.polynomial.polynomial.polyval(compression / fit["stroke_m"],
                                                        fit["monomial_coefficients_n"])
                # q increases AB; compression decreases; tau = -dU/dq = +F.
                max_virtual_work_error = max(max_virtual_work_error,
                                             abs((energies[1] - energies[0]) / (2 * eps) - force))
                assert binding["effort_sign"] == 1
                assert model.actuator_gear[model.actuator(name + "_gas_force").id, 0] == 1.
            mujoco.mj_resetDataKeyframe(model, data, 0)
            for name, value in q.items():
                data.qpos[model.joint(name).qposadr[0]] = value
            mujoco.mj_forward(model, data)
            jacobian = data.efc_J.reshape(data.nefc, model.nv)[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
            rank = int(np.linalg.matrix_rank(jacobian, tol=1e-8))
            min_rank = min(min_rank, rank)
            assert rank == 12
            q_samples.append({"hip_left_rad": hip, "inner_angle_deg": float(beta_deg),
                              "left_aux_motor_rad": q["LL_joint1"],
                              "spring_AB_m": .232 - compression, "constraint_rank": rank})
    assert max_closure < 2e-7 and max_frame_error < 1e-9 and max_angle_error < 1e-9
    assert max_virtual_work_error < 1e-5
    # Independently solve prescribed *motor* coordinates: equal hip/aux increments
    # should preserve the geometric knee angle and gas-spring length.
    neutral = spec["nominal_joint_pos"]
    coupled_angle_error = coupled_spring_error = 0.
    for delta in (-.08, .08):
        motors = {name: neutral[name] for name in ACTIVE}
        for name in ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1"):
            motors[name] += delta
        solved = solve_pose(spec, motors, neutral)
        for side in ("L", "R"):
            coupled_angle_error = max(coupled_angle_error,
                                     abs(inner_angle(spec, solved, side) - inner_angle(spec, neutral, side)))
            coupled_spring_error = max(coupled_spring_error,
                                      abs(solved[side + "_spring_slide"] - neutral[side + "_spring_slide"]))
    # Source pin origins are rounded and have sub-micrometre transverse offsets.
    # Preserve and report this error instead of silently moving those origins.
    zero_frames = fk(spec, {})
    coaxial_errors = {side: float(np.linalg.norm(np.cross(
        zero_frames[side * 2 + "_link1"][:3, 3] - zero_frames[side + "_link1"][:3, 3],
        zero_frames[side + "_link1"][:3, 2]))) for side in ("L", "R")}
    assert max(coaxial_errors.values()) < 1e-6
    assert coupled_angle_error < 1e-6 and coupled_spring_error < 1e-7
    return {"schema_version": 1, "passed": True, "model_sha256": digest(bundle / "robot.xml"),
            "source_sha256": digest(Path(__file__)),
            "export_limit_consistency_rad": limit_checks,
            "bundle_manifest_sha256": digest(bundle / "manifest.json"), "mujoco_version": mujoco.__version__,
            "topology": topology, "usd": usd_checks(bundle, spec),
            "knee_angle_definition": "thigh/shank pin-center vectors projected normal to common hinge axis",
            "source_zero_knee_inner_deg": {s: float(np.rad2deg(v)) for s, v in zero.items()},
            "domain_samples": len(q_samples), "domain": q_samples, "domain_constraint_min_rank": min_rank,
            "domain_max_closure_error_m": max_closure, "v5_mapped_visual_matrix_max_error": max_frame_error,
            "domain_max_knee_angle_error_rad": max_angle_error,
            "spring_compression_range_m": [min(compressions), max(compressions)],
            "spring_virtual_work_max_error_n": max_virtual_work_error,
            "equal_motor_increment_knee_error_rad": coupled_angle_error,
            "equal_motor_increment_spring_error_m": coupled_spring_error,
            "source_coaxial_pin_transverse_error_m": coaxial_errors,
            "sampled_hip_domain_is_hardware_safe_workspace": False,
            "not_validated": ["self_collision", "ground_clearance", "PhysX_dynamics", "hardware_limits_and_identification"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.bundle)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "domain"}, indent=2))
