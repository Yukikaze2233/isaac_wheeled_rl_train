"""Encoder-only wheel geometry from a sealed V6 closed-chain lookup.

The lookup is model-derived, not real-robot calibration. Inputs must already be
continuous encoder angles in the V6 model coordinate convention.
"""
from __future__ import annotations

import torch


class ClosedChainHeightModel:
    """Interpolate auxiliary-minus-hip geometry, then rotate about the hip axis."""

    def __init__(self, table, device):
        if table.get("schema") != "v6_closedchain_height_lookup_v1":
            raise ValueError("Unsupported closed-chain height lookup schema")
        if table.get("control_joint_names") != [
                "L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]:
            raise ValueError("Height lookup requires the six V6 model coordinates")
        self.sides = []
        for name in ("left", "right"):
            source = table["sides"][name]
            values = {key: torch.as_tensor(source[key], dtype=torch.float32, device=device)
                      for key in ("delta_rad", "wheel_center_b_m", "hip_origin_b_m",
                                  "hip_axis_b", "hip_reference_rad", "wheel_axis_b", "wheel_radius_m")}
            knots, points = values["delta_rad"], values["wheel_center_b_m"]
            if (knots.ndim != 1 or len(knots) < 2 or points.shape != (len(knots), 3)
                    or values["hip_origin_b_m"].shape != (3,)
                    or values["hip_axis_b"].shape != (3,) or values["wheel_axis_b"].shape != (3,)
                    or values["hip_reference_rad"].ndim != 0 or values["wheel_radius_m"].ndim != 0
                    or not all(bool(torch.isfinite(value).all()) for value in values.values())
                    or not bool((knots[1:] > knots[:-1]).all())
                    or not bool(values["wheel_radius_m"] > 0)
                    or not bool((values["hip_axis_b"].square().sum() - 1).abs() < 1e-5)
                    or not bool((values["wheel_axis_b"].square().sum() - 1).abs() < 1e-5)):
                raise ValueError(f"Invalid {name} closed-chain lookup")
            self.sides.append(values)
        self.device = self.sides[0]["delta_rad"].device
        self.wheel_radius_m = torch.stack([side["wheel_radius_m"] for side in self.sides])
        # V6 wheel/hip axes are parallel; wheel spin and common hip rotation do
        # not change these body-frame axes. The builder verifies that property.
        self.wheel_axles_b = torch.stack([side["wheel_axis_b"] for side in self.sides])

    def wheel_positions(self, q):
        """Return body-frame centers [N,2,3] and domain validity [N,2].

        Outside-domain differences are invalid, never extrapolated; whole-leg
        rotation remains continuous and has no artificial hip angle limit.
        """
        if q.ndim != 2 or q.shape[1] != 6 or q.device != self.device or not q.is_floating_point():
            raise ValueError("Expected floating V6 model joint positions [N,6] on lookup device")
        points, validity = [], []
        for side, hip, auxiliary in zip(self.sides, (0, 3), (1, 4)):
            knots = side["delta_rad"]
            delta = q[:, auxiliary] - q[:, hip]
            valid = (torch.isfinite(q[:, hip]) & torch.isfinite(delta)
                     & (delta >= knots[0] - 1e-6) & (delta <= knots[-1] + 1e-6))
            upper = torch.searchsorted(knots, delta.contiguous()).clamp(1, len(knots) - 1)
            lower = upper - 1
            blend = ((delta - knots[lower]) / (knots[upper] - knots[lower])).clamp(0., 1.)
            reference = side["wheel_center_b_m"]
            point = reference[lower] + blend[:, None] * (reference[upper] - reference[lower])
            vector = point - side["hip_origin_b_m"]
            axis = side["hip_axis_b"].to(dtype=vector.dtype).expand_as(vector)
            angle = q[:, hip] - side["hip_reference_rad"]
            cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
            rotated = (cosine * vector + sine * torch.linalg.cross(axis, vector)
                       + (1 - cosine) * axis * (axis * vector).sum(-1, keepdim=True))
            points.append(torch.where(valid[:, None], side["hip_origin_b_m"] + rotated, 0.))
            validity.append(valid)
        return torch.stack(points, dim=1), torch.stack(validity, dim=1)
