"""V7 declared thigh-shank inner coordinates from mapped passive native joints.

This adapter supports the reviewed closed-chain model coordinates. It does not
calibrate hardware, infer angles from motor labels, wrap continuous motors,
repair a broken chain, or select a reward/profile.
"""
from collections.abc import Mapping, Sequence
import math

import torch


_MODEL_KIND = "v6_gas_spring_closedchain_research"
_COORDINATES = "v6_download_link_frames_v1"
_CONTROL_NAMES = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
_SIDES = (("L_joint2", "L_link1", "L_link2", 1.),
          ("R_jonit2", "R_link1", "R_link2", -1.))


class RelativeKneeGeometry:
    """Bind the complete native joint order and return left/right inner radians.

    The supplied physical domain is an asset property, not an absolute motor
    angle limit. Joint origins and limits define the CAD zero; encoder/hardware
    zero certification remains outside this adapter. This is the reviewed CAD
    hinge/frame-angle coordinate, checked against pin vectors projected normal
    to the common hinge axis. The unprojected 3D pin-vector angle is a different
    geometric datum and is not calculated here.
    """

    def __init__(self, model_spec, manifest, native_joint_names):
        if not isinstance(model_spec, Mapping) or not isinstance(manifest, Mapping):
            raise ValueError("Relative knee geometry requires its explicit model and manifest")
        if any(value.get("model_kind") != _MODEL_KIND
               or value.get("coordinate_schema") != _COORDINATES
               or value.get("control_frame") != "Xforward_Yleft_Zup"
               for value in (model_spec, manifest)):
            raise ValueError("Relative knee geometry does not support unknown asset coordinates")
        if tuple(manifest.get("control_joint_names", ())) != _CONTROL_NAMES:
            raise ValueError("Relative knee geometry requires the named continuous motor roles")
        if manifest.get("knee_angle_definition") != "pin_center_vectors_projected_normal_to_common_hinge_axis":
            raise ValueError("Relative knee geometry requires the physical pin-center inner-angle definition")
        physical = manifest.get("knee_inner_limits_deg")
        if (not isinstance(physical, Sequence) or len(physical) != 2
                or any(type(value) not in (int, float) for value in physical)
                or not all(math.isfinite(value) for value in physical)
                or not 0. < physical[0] < physical[1] < 180.):
            raise ValueError("Relative knee geometry requires a finite physical inner-angle interval")
        self.inner_limits_rad = tuple(math.radians(value) for value in physical)
        span = self.inner_limits_rad[1] - self.inner_limits_rad[0]
        joints = model_spec.get("joints")
        if not isinstance(joints, Sequence) or not joints or any(not isinstance(joint, Mapping) for joint in joints):
            raise ValueError("Relative knee geometry requires the complete named joint tree")
        names = [joint.get("name") for joint in joints]
        declared = manifest.get("tree_joint_names")
        native = native_joint_names
        if (not isinstance(native, Sequence) or isinstance(native, (str, bytes))
                or not isinstance(declared, Sequence) or isinstance(declared, (str, bytes))
                or any(not isinstance(name, str) or not name for name in (*names, *native, *declared))
                or len(set(names)) != len(names) or len(set(native)) != len(native)
                or len(set(declared)) != len(declared)
                or set(names) != set(declared) or set(native) != set(declared)):
            raise ValueError("Relative knee geometry requires the complete unique native joint order")
        by_name = dict(zip(names, joints))
        for name in ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1"):
            joint = by_name.get(name, {})
            if joint.get("type") != "continuous" or joint.get("parent") != "base_link":
                raise ValueError("Relative knee geometry preserves four continuous root-driven axes")
        zeros, indices = [], []
        for name, thigh, shank, sign in _SIDES:
            joint = by_name.get(name, {})
            if (joint.get("type") != "revolute" or joint.get("parent") != thigh
                    or joint.get("child") != shank or name in _CONTROL_NAMES):
                raise ValueError("Relative knee geometry requires the passive thigh-to-shank hinges")
            axis = joint.get("axis")
            if (not isinstance(axis, Sequence) or len(axis) != 3
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in axis)
                    or any(abs(value - expected) > 1e-6 for value, expected in zip(axis, (0., 0., sign)))):
                raise ValueError("Relative knee geometry hinge axis/sign differs from the model coordinates")
            origin = joint.get("origin")
            cosine, sine = math.cos(-2.3573), math.sin(-2.3573)
            expected_origin = ((cosine, -sine, 0., 0.), (sine, cosine, 0., .21),
                               (0., 0., 1., 0.), (0., 0., 0., 1.))
            if (not isinstance(origin, Sequence) or len(origin) != 4
                    or any(not isinstance(row, Sequence) or len(row) != 4 for row in origin)
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or abs(value - expected) > 1e-8
                           for row, expected_row in zip(origin, expected_origin)
                           for value, expected in zip(row, expected_row))):
                raise ValueError("Relative knee geometry hinge origin differs from the reviewed CAD coordinates")
            limits = joint.get("limit", {})
            try:
                lower, upper = float(limits["lower"]), float(limits["upper"])
            except (TypeError, KeyError, ValueError):
                raise ValueError("Relative knee geometry requires explicit passive joint limits") from None
            if (not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper
                    or not math.isclose(upper - lower, span, abs_tol=1e-10, rel_tol=0.)):
                raise ValueError("Passive joint limit span differs from the physical inner-angle interval")
            zero = self.inner_limits_rad[0] - lower if sign == 1. else self.inner_limits_rad[0] + upper
            # The known coordinate schema uses this CAD hinge zero. Limits are
            # still the source of the returned zero, independently checked here
            # against the builder's origin convention; shifted limits are not a
            # hardware recalibration or another supported asset.
            if not math.isclose(zero, math.pi - 2.3573, abs_tol=1e-10, rel_tol=0.):
                raise ValueError("Passive joint zero differs from the reviewed CAD coordinates")
            zeros.append(zero)
            indices.append(native.index(name))
        self.native_joint_names = tuple(native)
        self.joint_indices = tuple(indices)
        self.knee_zero_rad = tuple(zeros)

    def inner_angles_rad(self, native_joint_position):
        """Read actual full native q; nonfinite/overtravel values remain visible."""
        q = native_joint_position
        if (not isinstance(q, torch.Tensor) or q.ndim != 2 or q.shape[0] == 0
                or q.shape[1] != len(self.native_joint_names)
                or q.dtype not in (torch.float32, torch.float64)):
            raise ValueError("Relative knee geometry requires full native float32/float64 joint rows")
        left, right = self.joint_indices
        return torch.stack((q[:, left] + self.knee_zero_rad[0],
                            -q[:, right] + self.knee_zero_rad[1]), dim=-1)
