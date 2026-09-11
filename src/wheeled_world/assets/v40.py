"""Canonical seven-body V40 URDF; explicit efforts, no importer or actuator PD.

Isaac Lab v2.3.0 UrdfConverterCfg supports JointDriveCfg(target_type="none").
All physical limits passed here must originate in the validated V40 contract.
"""
from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg


def normalize_v40_usd_joint_limits(stage, env_prim_paths, joint_contract, *, usd_physics=None):
    """Restore audited URDF continuous semantics after cloning, BEFORE physics starts.

    The source's continuous joints carry placeholder +/-3.14 limit attributes.
    The importer can turn these into revolute hard stops despite the URDF type.
    Author stronger stage opinions, not edits to hash-bound URDF/cache/evidence.
    Knees are checked in radians but never written. Validate all clones first.
    """
    if usd_physics is None:
        from pxr import UsdPhysics
        usd_physics = UsdPhysics
    continuous = set(joint_contract["continuous"])
    knees = joint_contract["knee_hard_limits"]
    expected = continuous | set(knees)
    if (continuous != {"L_joint1", "R_joint1", "L_joint3", "R_joint3"}
            or set(knees) != {"L_joint2", "R_jonit2"}):
        raise ValueError("expected V40 continuous hips/wheels and two bounded knees")
    paths = list(env_prim_paths)
    if not paths or len(set(paths)) != len(paths) or any(
        not isinstance(path, str) or not path.startswith("/") or path.endswith("/")
        or ".." in path.split("/") for path in paths
    ):
        raise ValueError("unique absolute environment USD paths are required")
    all_joints = [prim for prim in stage.Traverse() if prim.IsA(usd_physics.Joint)]
    plans = []
    for env_path in paths:
        root_path = env_path + "/Robot"
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise ValueError(f"missing USD robot clone: {root_path}")
        joints = {}
        for prim in all_joints:
            if not str(prim.GetPath()).startswith(root_path + "/"):
                continue
            name = prim.GetName()
            if name not in expected or name in joints or not prim.IsA(usd_physics.RevoluteJoint):
                raise ValueError(f"unknown/ambiguous/non-revolute V40 USD joint: {prim.GetPath()}")
            # D6 LimitAPI is not the typed revolute lowerLimit/upperLimit schema.
            # Refuse an unexpected additional constraint instead of silently rewriting it.
            if any(str(api).startswith("PhysicsLimitAPI:") for api in prim.GetAppliedSchemas()):
                raise ValueError(f"unexpected additional USD LimitAPI at {prim.GetPath()}")
            joint = usd_physics.RevoluteJoint(prim)
            if name in knees:
                actual = [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
                if any(value is None or not math.isfinite(value)
                       or not math.isclose(math.radians(value), bound, abs_tol=1e-6, rel_tol=0)
                       for value, bound in zip(actual, knees[name], strict=True)):
                    raise ValueError(f"V40 USD knee limit mismatch at {prim.GetPath()}: {actual} degrees")
            joints[name] = joint
        if set(joints) != expected:
            raise ValueError(f"missing V40 USD joints in {root_path}: {expected - set(joints)}")
        plans.append((root_path, joints))
    for root_path, joints in plans:
        for name in sorted(continuous):
            joint = joints[name]
            if (joint.CreateLowerLimitAttr().Set(-math.inf) is False
                    or joint.CreateUpperLimitAttr().Set(math.inf) is False):
                raise RuntimeError(f"failed writing continuous USD limits: {root_path}/{name}")
    # Read composed values after ALL edits, including the untouched knees.
    for root_path, joints in plans:
        for name, joint in joints.items():
            actual = [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
            if name in continuous:
                matches = actual == [-math.inf, math.inf]
            else:
                matches = all(value is not None and math.isfinite(value)
                              and math.isclose(math.radians(value), bound, abs_tol=1e-6, rel_tol=0)
                              for value, bound in zip(actual, knees[name], strict=True))
            if not matches:
                raise RuntimeError(f"V40 USD limit readback mismatch: {root_path}/{name}: {actual} degrees")
    return {"env_count": len(paths), "schema": "UsdPhysics.RevoluteJoint",
            "continuous_joints": sorted(continuous), "continuous_limits_deg": ["-inf", "+inf"],
            "scope": "composed USD limits; actual solver limits checked after physics initialization"}


def validate_v40_physx_joint_limits(robot, joint_contract, *, num_envs):
    """Read the actual solver tensor in imported name order, not cached soft limits.

    Check every clone. PhysX can encode an unbounded PxReal range as the exact
    float32 extrema instead of IEEE infinities. Accept those two paired encodings
    only, not arbitrary large finite bounds, mixed encodings or one-sided limits.
    """
    import torch

    names = list(robot.joint_names)
    expected = list(joint_contract["action_order"])
    if len(names) != 6 or len(set(names)) != 6 or set(names) != set(expected):
        raise RuntimeError(f"V40 PhysX joint name mismatch: {names}")
    limits = robot.root_physx_view.get_dof_limits()
    if num_envs < 1 or tuple(limits.shape) != (num_envs, 6, 2):
        raise RuntimeError(f"V40 PhysX DOF limit shape mismatch: {tuple(limits.shape)}")
    float32_max = torch.finfo(torch.float32).max
    report, encodings = {}, {}
    for name in expected:
        actual = limits[:, names.index(name), :]
        if name in joint_contract["continuous"]:
            ieee_unbounded = torch.isneginf(actual[:, 0]) & torch.isposinf(actual[:, 1])
            # Exact native sentinel equality, deliberately no tolerance/size threshold.
            native_unbounded = (actual[:, 0] == -float32_max) & (actual[:, 1] == float32_max)
            valid = ieee_unbounded | native_unbounded
            encoding = "ieee_infinity" if bool(ieee_unbounded[0]) else "physx_float32_extrema"
        else:
            bound = actual.new_tensor(joint_contract["knee_hard_limits"][name])
            valid = (torch.isfinite(actual) & torch.isclose(actual, bound, atol=1e-6, rtol=0)).all(-1)
            encoding = "finite_knee"
        if not bool(valid.all()):
            env_id = int((~valid).nonzero(as_tuple=False)[0, 0])
            raise RuntimeError(
                f"V40 PhysX DOF limit mismatch: env {env_id}, {name}: {actual[env_id].tolist()} rad; "
                "expected continuous (-inf, +inf) or exact (-FLT_MAX, +FLT_MAX), "
                "or the contract knee bounds. Check the USD import boundary."
            )
        # Preserve finite raw readback, including FLT_MAX; only IEEE infinities need
        # string encoding for strict JSON. Labels describe env 0, not other clones.
        report[name] = ["-inf", "+inf"] if encoding == "ieee_infinity" else actual[0].tolist()
        encodings[name] = encoding
    return {"env_count": num_envs, "shape": list(limits.shape), "limits_rad_env0": report,
            "limit_encoding_env0": encodings,
            "scope": "root_physx_view.get_dof_limits(), all environments checked; not a multi-turn motion test"}


def apply_approved_collision_filters(
    stage, env_prim_paths, manifest, *, research_approval=None, raw_manifest=None, usd_physics=None,
):
    """Author only reviewed adjacent rigid-body pairs, independently for each USD clone.

    Validate the ENTIRE plan before any USD edits. The optional schema argument is
    a test seam for a pure fake stage; production imports pxr only after approval.
    Research approval/raw dictionaries must come from validate_asset hash auditing;
    approval permits only the named equivalent model, not a repair of raw geometry.
    Relationship readback is not evidence of the PhysX collision implementation.
    """
    adjacency = {
        "L_joint1": ("base_link", "L_link1"), "L_joint2": ("L_link1", "L_link2"),
        "L_joint3": ("L_link2", "L_link3"), "R_joint1": ("base_link", "R_link1"),
        "R_jonit2": ("R_link1", "R_link2"), "R_joint3": ("R_link2", "R_link3"),
    }
    from wheeled_tasks.v40.contract import validate_filter_policy
    # Shared validator keeps raw-supported and explicitly approved research scopes
    # separate, preserving unresolved source material flags for the latter.
    validate_filter_policy(manifest, research_approval=research_approval, raw_manifest=raw_manifest)
    paths = list(env_prim_paths)
    if not paths or len(set(paths)) != len(paths) or any(
        not isinstance(path, str) or not path.startswith("/") or path.endswith("/")
        or ".." in path.split("/") for path in paths
    ):
        raise ValueError("unique absolute environment USD paths are required")
    if usd_physics is None:
        from pxr import UsdPhysics
        usd_physics = UsdPhysics
    body_names = {body for pair in adjacency.values() for body in pair}
    all_prims = list(stage.Traverse())
    plans = []
    for env_path in paths:
        root_path = env_path + "/Robot"
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise ValueError(f"missing USD robot clone: {root_path}; Fabric-only clones are not supported")
        robot_prims = [prim for prim in all_prims if str(prim.GetPath()) == root_path
                       or str(prim.GetPath()).startswith(root_path + "/")]
        bodies = {}
        for prim in robot_prims:
            if prim.HasAPI(usd_physics.RigidBodyAPI):
                name = prim.GetName()
                if name not in body_names or name in bodies:
                    raise ValueError(f"unknown/ambiguous rigid-body name in {root_path}: {name}")
                bodies[name] = prim
        if set(bodies) != body_names:
            raise ValueError(f"missing named rigid bodies in {root_path}: {body_names - set(bodies)}")
        targets = {name: set() for name in body_names}
        for body1, body2 in adjacency.values():
            # Explicit reciprocal targets make the readback unambiguous (12 edges / 6 pairs).
            targets[body1].add(str(bodies[body2].GetPath()))
            targets[body2].add(str(bodies[body1].GetPath()))
        for prim in robot_prims:
            relation = usd_physics.FilteredPairsAPI(prim).GetFilteredPairsRel()
            existing = {str(path) for path in relation.GetTargets()} if relation else set()
            allowed = targets.get(prim.GetName(), set()) if prim.HasAPI(usd_physics.RigidBodyAPI) else set()
            if not existing.issubset(allowed):
                raise ValueError(f"unreviewed existing collision filter at {prim.GetPath()}: {existing - allowed}")
        plans.append((env_path, bodies, targets))
    reports = []
    for env_path, bodies, targets in plans:
        for name, prim in bodies.items():
            relation = usd_physics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
            target_paths = [stage.GetPrimAtPath(path).GetPath() for path in sorted(targets[name])]
            if relation.SetTargets(target_paths) is False:
                raise RuntimeError(f"failed writing collision filter at {prim.GetPath()}")
        count = 0
        for name, prim in bodies.items():
            readback = usd_physics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
            if {str(path) for path in readback} != targets[name] or len(readback) != len(targets[name]):
                raise RuntimeError(f"collision filter readback mismatch at {prim.GetPath()}")
            count += len(readback)
        reports.append({"env_path": env_path, "pair_count": len(adjacency), "relationship_target_count": count})
    return {"envs": reports, "scope": "USD relationship authoring/readback only; PhysX effects unverified"}


def make_v40_articulation(
    *, urdf_path: str | Path, joint_names: list[str], nominal_positions: list[float],
    effort_limits: list[float], armatures: list[float],
    nominal_base_height: float, asset_manifest_sha256: str,
    usd_cache_dir: str | Path | None = None,
) -> ArticulationCfg:
    """Keep the importer cache tied to the asset manifest, not a stale old USD."""
    vectors = (nominal_positions, effort_limits, armatures)
    if len(joint_names) != 6 or len(set(joint_names)) != 6 or any(len(v) != 6 for v in vectors):
        raise ValueError("V40 requires six unique named joints and six values per parameter")
    if not all(math.isfinite(float(x)) for vector in vectors for x in vector):
        raise ValueError("non-finite actuator parameters")
    if any(x <= 0 for x in effort_limits) or any(x < 0 for x in armatures):
        raise ValueError("effort limits must be positive; armature must be nonnegative")
    if len(asset_manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in asset_manifest_sha256):
        raise ValueError("invalid asset manifest digest")
    # Separate actuator groups avoid any dependency on regex resolution order.
    actuators = {
        f"effort_{i}": IdealPDActuatorCfg(
            joint_names_expr=[name], stiffness=0.0, damping=0.0,
            effort_limit=effort_limits[i], effort_limit_sim=effort_limits[i],
            # Disable the URDF's placeholder 1 rad/s solver clamp. This large
            # solver bound is NOT a motor rating; core enforces the wheel curve.
            velocity_limit_sim=1.0e9,
            armature=armatures[i], friction=0.0,
        ) for i, name in enumerate(joint_names)
    }
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(Path(urdf_path).resolve()),
            usd_dir=str(Path(usd_cache_dir).resolve() if usd_cache_dir is not None else
                        Path(urdf_path).resolve().parents[2] / "logs" / "v40_usd_cache" / asset_manifest_sha256),
            usd_file_name="robot.usd",
            force_usd_conversion=True,
            fix_base=False,
            root_link_name="base_link",
            merge_fixed_joints=False,
            self_collision=True,
            collision_from_visuals=False,
            collider_type="convex_hull",
            replace_cylinders_with_capsules=False,
            activate_contact_sensors=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="none",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, nominal_base_height),
            # The URDF is already X-forward/Y-left/Z-up. Keep the default quaternion.
            joint_pos=dict(zip(joint_names, nominal_positions)),
            joint_vel={name: 0.0 for name in joint_names},
        ),
        actuators=actuators,
        soft_joint_pos_limit_factor=1.0,
    )
