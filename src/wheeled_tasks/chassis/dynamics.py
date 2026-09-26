"""Startup rigid-body variation with explicit nominal and fixed-case controls."""
from __future__ import annotations

import hashlib
import math

import torch


class RigidBodyRandomization:
    """Sample one stationary mechanical variant per environment, before reset.

    Masses are in kg, COM poses are body-local xyz[m]/xyzw, and inertias retain
    the backend's complete nine-component representation. A positive scalar
    scales the entire inertia tensor, preserving its symmetry and definiteness.
    """

    def __init__(self, masses, inertias, coms, body_names, config, seed, profiles=None, enabled_mask=None):
        allowed = {"enabled", "sampling", "enabled_fraction", "base_body", "wheel_bodies",
                   "base_mass_scale", "leg_mass_scale", "wheel_mass_scale", "inertia_scale",
                   "base_com_offset_m", "source_reference"}
        if set(config) - allowed:
            raise ValueError("Unknown rigid-body randomization configuration field")
        self.cfg = dict(config)
        self.seed = int(seed)
        self.count, self.body_count = masses.shape
        self.device = masses.device
        if (inertias.shape != (self.count, self.body_count, 9)
                or coms.shape != (self.count, self.body_count, 7)
                or len(body_names) != self.body_count
                or not bool(torch.isfinite(masses).all() & (masses > 0).all())
                or not bool(torch.isfinite(inertias).all() & torch.isfinite(coms).all())):
            raise ValueError("Invalid nominal rigid-body property arrays")
        if config.get("sampling", "startup") != "startup":
            raise ValueError("Rigid-body variation is startup-only; no mid-episode mass changes")
        self.base_id = body_names.index(config.get("base_body", "base_link"))
        self.wheel_ids = [body_names.index(name) for name in config.get("wheel_bodies", ["L_link3", "R_link3"])]
        if len(set([self.base_id, *self.wheel_ids])) != 3:
            raise ValueError("Base and two wheel bodies must be distinct")
        self.other_ids = [i for i in range(self.body_count) if i not in [self.base_id, *self.wheel_ids]]
        self.ranges = {name: self._range(config.get(name, [1., 1.]), positive=True)
                       for name in ("base_mass_scale", "leg_mass_scale", "wheel_mass_scale", "inertia_scale")}
        if any(not lo <= 1. <= hi for lo, hi in self.ranges.values()):
            raise ValueError("Declared scale domains must include the nominal control")
        com_ranges = config.get("base_com_offset_m", {})
        if not isinstance(com_ranges, dict) or set(com_ranges) - {"x", "y", "z"}:
            raise ValueError("COM offsets must use body-local x/y/z")
        self.com_ranges = [self._range(com_ranges.get(axis, [0., 0.])) for axis in ("x", "y", "z")]
        if any(not lo <= 0. <= hi for lo, hi in self.com_ranges):
            raise ValueError("Declared COM domains must include zero offset")
        fraction = config.get("enabled_fraction", .5)
        if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Dynamics enabled fraction must be in [0,1]")
        if profiles is not None and len(profiles) != self.count:
            raise ValueError("One explicit dynamics profile is required per evaluation environment")

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        low = masses.new_full((self.body_count,), self.ranges["leg_mass_scale"][0])
        high = masses.new_full((self.body_count,), self.ranges["leg_mass_scale"][1])
        for ids, name in (([self.base_id], "base_mass_scale"), (self.wheel_ids, "wheel_mass_scale")):
            low[ids], high[ids] = self.ranges[name]
        self.enabled = torch.rand(self.count, generator=generator, device=self.device) < fraction
        if enabled_mask is not None:
            if enabled_mask.shape != self.enabled.shape:
                raise ValueError("Domain lane mask has the wrong shape")
            self.enabled = enabled_mask.to(device=self.device, dtype=torch.bool).clone()
        self.mass_scale = low + (high - low) * torch.rand(masses.shape, generator=generator, device=self.device)
        inertia_low, inertia_high = self.ranges["inertia_scale"]
        self.inertia_scale = inertia_low + (inertia_high - inertia_low) * torch.rand(
            masses.shape, generator=generator, device=self.device)
        lower = masses.new_tensor([r[0] for r in self.com_ranges])
        upper = masses.new_tensor([r[1] for r in self.com_ranges])
        self.com_offset = lower + (upper - lower) * torch.rand(
            (self.count, 3), generator=generator, device=self.device)
        if profiles is not None:
            self.enabled.zero_()
            self.mass_scale.fill_(1.)
            self.inertia_scale.fill_(1.)
            self.com_offset.zero_()
            for row, profile in enumerate(profiles):
                if profile is not None:
                    self._set_profile(row, profile)
        self.mass_scale = torch.where(self.enabled[:, None], self.mass_scale, 1.)
        self.inertia_scale = torch.where(self.enabled[:, None], self.inertia_scale, 1.)
        self.com_offset = torch.where(self.enabled[:, None], self.com_offset, 0.)
        self.masses = masses * self.mass_scale
        self.inertias = inertias * (self.mass_scale * self.inertia_scale)[..., None]
        self.coms = coms.clone()
        self.coms[:, self.base_id, :3] += self.com_offset
        self.total_mass_bounds = [float((masses[0] * bound).sum()) for bound in (low, high)]
        self._metrics = {
            "/domain/dynamics_enabled_fraction": self.enabled.float().mean(),
            "/domain/total_mass_kg_min": self.masses.sum(-1).min(),
            "/domain/total_mass_kg_max": self.masses.sum(-1).max(),
            "/domain/base_mass_scale_mean": self.mass_scale[:, self.base_id].mean(),
            "/domain/base_com_offset_x_abs_max_m": self.com_offset[:, 0].abs().max(),
            "/domain/base_com_offset_y_abs_max_m": self.com_offset[:, 1].abs().max(),
            "/domain/base_com_offset_z_abs_max_m": self.com_offset[:, 2].abs().max(),
        }

    @staticmethod
    def _range(value, positive=False):
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or not all(math.isfinite(v) for v in value)
                or value[0] > value[1] or (positive and value[0] <= 0)):
            raise ValueError("Invalid rigid-body randomization range")
        return tuple(value)

    def _set_profile(self, row, profile):
        allowed = {"base_mass_scale", "leg_mass_scale", "wheel_mass_scale", "inertia_scale", "base_com_offset_m"}
        if not isinstance(profile, dict) or set(profile) - allowed:
            raise ValueError("Unknown evaluation dynamics profile field")
        for ids, name in ((self.other_ids, "leg_mass_scale"), ([self.base_id], "base_mass_scale"),
                          (self.wheel_ids, "wheel_mass_scale")):
            value = profile.get(name, 1.)
            lo, hi = self.ranges[name]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Evaluation {name} is outside the declared domain")
            self.mass_scale[row, ids] = value
        value = profile.get("inertia_scale", 1.)
        lo, hi = self.ranges["inertia_scale"]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError("Evaluation inertia scale is outside the declared domain")
        self.inertia_scale[row] = value
        offset = profile.get("base_com_offset_m", [0., 0., 0.])
        if len(offset) != 3 or any(not math.isfinite(v) or not lo <= v <= hi
                                   for v, (lo, hi) in zip(offset, self.com_ranges)):
            raise ValueError("Evaluation COM offset is outside the declared domain")
        self.com_offset[row] = self.com_offset.new_tensor(offset)
        self.enabled[row] = bool(profile)

    def apply(self, robot):
        """Use the asset setters so both backend and cached model properties update."""
        import warp as wp

        robot.set_masses_index(masses=self.masses)
        robot.set_coms_index(coms=self.coms)
        # PhysX diagonalizes a body-frame inertia and may permute its principal
        # axes. Restoring the old COM quaternion afterwards would rotate the new
        # principal moments into the wrong axes, changing the physical tensor.
        robot.set_inertias_index(inertias=self.inertias)
        resolved_coms = wp.to_torch(robot.root_view.get_coms()).to(self.device).clone()
        robot.set_coms_index(coms=resolved_coms)

    def metrics(self):
        return self._metrics

    def summary(self):
        digest = hashlib.sha256()
        for values in (self.masses, self.inertias, self.coms):
            digest.update(values.detach().cpu().contiguous().numpy().tobytes())
        return {
            "sampling": "startup_per_environment_episode_static", "seed": self.seed,
            "configuration": self.cfg, "sampled_properties_sha256": digest.hexdigest(),
            "total_mass_bounds_kg": self.total_mass_bounds,
            "metrics": {name: float(value) for name, value in self._metrics.items()},
            "inertia_mass_scaling": "full_tensor_scaled_with_mass_then_optional_independent_scalar",
            "com_frame": "body_local_translation_with_backend_principal_axes_relabeling",
        }
