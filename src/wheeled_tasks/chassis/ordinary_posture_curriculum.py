"""Opt-in ordinary posture births and the existing per-lane velocity pushes.

The actor clock changes amplitudes only. Episode reset samples independent
timers; no script state is overwritten and no simulator/controller lives here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from .nonstandard_start import support_height_for_footprint


ORDINARY_POOLS = frozenset({"stand_anchors", "stand_continuous", "height",
    "basic_translation", "basic_transitions", "basic_yaw", "high_motion",
    "high_transitions", "release_stop", "high_translation",
    "surface_height_joint", "surface_traversal", "untriggered_steps"})


def posture_settings(origin_actor_update, enabled=True):
    if type(origin_actor_update) is not int or origin_actor_update < 0:
        raise ValueError("Ordinary posture origin must be a nonnegative integer")
    if type(enabled) is not bool:
        raise ValueError("Ordinary posture enabled must be bool")
    return {"version": "ordinary_posture_v1", "enabled": enabled,
        "training_only": True, "clock": "domain_actor_updates",
        "origin_actor_update": origin_actor_update,
        "ramp": {"start_actor_update": 1000, "end_actor_update": 3000},
        "birth": {"probability": .5, "start_roll_pitch_rad": .03,
            "end_roll_pitch_rad": .15},
        "push": {"enabled": True, "start_speed_m_s": .05,
            "end_speed_m_s": .25, "interval_seconds": [5., 10.]}}


def validate_posture_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Ordinary posture needs its complete v1 settings")
    expected = posture_settings(settings.get("origin_actor_update", 0),
        settings.get("enabled", False))
    if settings != expected:
        raise ValueError("Ordinary posture settings differ from the frozen v1 curriculum")


def runtime_expectation(settings, enabled=None):
    validate_posture_settings(settings)
    if enabled is not None and type(enabled) is not bool:
        raise ValueError("Ordinary posture runtime enabled must be bool")
    return {"enabled": settings["enabled"] if enabled is None else enabled,
        "version": settings["version"], "clock": settings["clock"],
        "origin_actor_update": settings["origin_actor_update"],
        "settings": deepcopy(settings), "state_writes": "episode_initial_reset_only",
        "birth_scope": "ordinary_flat_and_terrain_without_requests",
        "push_application": "native_planar_velocity_delta",
        "push_clock": "independent_per_lane_policy_seconds",
        "push_ownership": "ground_actor_only_without_passive_hold"}


def ordinary_eligibility(config, groups, kinds, device, *, recovery=None, spin=None):
    """Whitelist ordinary pools; a flat-looking request row is still excluded."""
    membership, specs = config["behavior_pool_membership"], config["skill_specs"]
    selected = []
    for group, terrain in zip(groups, kinds):
        spec = specs[group]
        requests = (spec.get("mode", 0) == 4 or spec.get("request_events")
            or spec.get("command_sequence") or spec.get("recovery_pose")
            or spec.get("kind") in {"jump", "airborne", "landing", "recovery", "spin"})
        selected.append(membership[group] in ORDINARY_POOLS
            and terrain != "jump" and not requests)
    eligible = torch.tensor(selected, device=device, dtype=torch.bool)
    for excluded in (recovery, spin):
        if excluded is not None:
            eligible &= ~excluded
    return eligible


def rotate_vectors(quaternion, vectors):
    """XY Z W quaternions; vector dimensions may include collision primitives."""
    q = quaternion
    while q.ndim < vectors.ndim:
        q = q.unsqueeze(-2)
    xyz = q[..., :3].expand_as(vectors)
    return vectors + 2 * torch.linalg.cross(xyz,
        torch.linalg.cross(xyz, vectors) + q[..., 3:4] * vectors)


class OrdinaryPostureCurriculum:
    def __init__(self, count, device, settings, eligible, *, evaluation=False):
        validate_posture_settings(settings)
        if type(count) is not int or count < 1:
            raise ValueError("Ordinary posture needs a positive lane count")
        self.device, self.count = torch.device(device), count
        if eligible.shape != (count,) or eligible.dtype != torch.bool or eligible.device != self.device:
            raise ValueError("Ordinary posture needs a boolean native lane eligibility mask")
        self.settings = deepcopy(settings)
        self.enabled = settings["enabled"] and not evaluation
        self.eligible = eligible.clone() & self.enabled
        self.selected = torch.zeros_like(self.eligible)
        self.angles = torch.zeros(count, 2, device=device)
        self.birth_population = int(self.eligible.sum())
        self.birth_selected_count = self.birth_rejected_count = self.push_events = 0
        self.birth_abs_angle_max_rad = self.push_velocity_delta_max_m_s = 0.
        self.timer_samples = torch.zeros(count, device=device)
        self.timer_metrics = {"push_clock_min_s": 0., "push_clock_max_s": 0., "push_clock_std_s": 0.}
        self.actor_update = settings["origin_actor_update"]
        self.geometry_bound = False

    def _ids(self, ids):
        return torch.as_tensor(ids, device=self.device, dtype=torch.long)

    def amplitudes(self, actor_update):
        if type(actor_update) is not int or actor_update < 0:
            raise ValueError("Ordinary posture uses a nonnegative integer actor clock")
        self.actor_update = actor_update
        relative = max(0, actor_update - self.settings["origin_actor_update"])
        ramp = self.settings["ramp"]
        alpha = min(1., max(0., (relative - ramp["start_actor_update"])
            / (ramp["end_actor_update"] - ramp["start_actor_update"])))
        birth, push = self.settings["birth"], self.settings["push"]
        angle = birth["start_roll_pitch_rad"] + alpha * (
            birth["end_roll_pitch_rad"] - birth["start_roll_pitch_rad"])
        speed = push["start_speed_m_s"] + alpha * (
            push["end_speed_m_s"] - push["start_speed_m_s"])
        return relative, angle, speed

    def reset(self, ids, actor_update, *, generator):
        ids = self._ids(ids)
        _, cap, _ = self.amplitudes(actor_update)
        self.selected[ids] = False
        self.angles[ids] = 0.
        selected_ids = ids[self.eligible[ids]]
        if not len(selected_ids):
            return
        draw = torch.rand((len(selected_ids), 3), device=self.device, generator=generator)
        selected_ids = selected_ids[draw[:, 0] < self.settings["birth"]["probability"]]
        self.selected[selected_ids] = True
        self.angles[selected_ids] = (2 * draw[draw[:, 0] < self.settings["birth"]["probability"], 1:] - 1.) * cap

    def sample_intervals(self, ids, *, generator):
        ids = self._ids(ids)
        lo, hi = self.settings["push"]["interval_seconds"]
        intervals = lo + (hi - lo) * torch.rand(len(ids), device=self.device, generator=generator)
        self.timer_samples[ids] = intervals
        timers = self.timer_samples[self.eligible]
        if timers.numel():
            self.timer_metrics = {"push_clock_min_s": float(timers.amin()),
                "push_clock_max_s": float(timers.amax()),
                "push_clock_std_s": float(timers.std(unbiased=False))}
        return intervals

    def initial_orientation(self, ids, root):
        ids = self._ids(ids)
        selected = self.selected[ids]
        roll, pitch = (self.angles[ids[selected]] * .5).unbind(-1)
        root[selected, 3:7] = torch.stack((roll.sin() * pitch.cos(),
            roll.cos() * pitch.sin(), -roll.sin() * pitch.sin(),
            roll.cos() * pitch.cos()), -1)

    def bind_geometry(self, directory, manifest, joint_names, nonstandard=None):
        """Cache sealed FK collision bounds once; never solve or rebuild a leg.

        Only the unchanged nominal and already authenticated closed-chain bank
        coordinates can be used by this reset-only curriculum.
        """
        directory = Path(directory)
        path = directory / "tools/v6_mechanism.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files_sha256"]["tools/v6_mechanism.py"]:
            raise ValueError("Ordinary posture FK differs from the sealed asset")
        spec_path = directory / "model_spec.json"
        if hashlib.sha256(spec_path.read_bytes()).hexdigest() != manifest["files_sha256"]["model_spec.json"]:
            raise ValueError("Ordinary posture geometry differs from the sealed asset")
        import json
        spec = json.loads(spec_path.read_bytes())
        loader = importlib.util.spec_from_file_location("ordinary_posture_sealed_fk", path)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        poses = [[manifest["nominal_joint_pos"][name] for name in joint_names]]
        if nonstandard is not None:
            poses += nonstandard.joint_positions.detach().cpu().tolist()
        corners, wheels, axes = [], [], []
        mesh_vertices = {}
        for body in spec["bodies"]:
            for collision in body["collisions"]:
                if collision["type"] == "mesh":
                    filename = collision["file"]
                    mesh_path = directory / filename
                    if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != manifest["files_sha256"][filename]:
                        raise ValueError("Ordinary posture collision mesh differs from the sealed asset")
                    mesh_vertices[filename] = np.asarray([[float(x) for x in line.split()[1:4]]
                        for line in mesh_path.read_text().splitlines() if line.startswith("v ")])
        self.wheel_radius, self.wheel_halfwidth = None, None
        for pose in poses:
            frames = module.fk(spec, dict(zip(joint_names, pose)))
            boxes, wheel_points, wheel_axes, radii, halfwidths = [], [], [], [], []
            for body in spec["bodies"]:
                for collision in body["collisions"]:
                    transform = frames[body["name"]] @ np.asarray(collision["origin"])
                    if body["name"] in ("L_link3", "R_link3"):
                        if collision["type"] != "cylinder":
                            raise ValueError("Ordinary posture requires the sealed wheel cylinders")
                        wheel_points.append(transform[:3, 3]); wheel_axes.append(transform[:3, 2])
                        radii.append(collision["radius"]); halfwidths.append(collision["length"] * .5)
                    elif collision["type"] == "mesh":
                        vertices = mesh_vertices[collision["file"]] @ transform[:3, :3].T + transform[:3, 3]
                        lo, hi = vertices.min(0), vertices.max(0)
                        boxes.append(np.asarray([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
                    else:
                        raise ValueError("Unsupported ordinary posture collision primitive")
            if len(wheel_points) != 2 or not boxes:
                raise ValueError("Ordinary posture requires both wheels and body collision bounds")
            corners.append(boxes); wheels.append(wheel_points); axes.append(wheel_axes)
            self.wheel_radius = torch.tensor(radii, device=self.device)
            self.wheel_halfwidth = torch.tensor(halfwidths, device=self.device)
        self.profile_positions = torch.tensor(poses, device=self.device)
        self.collision_corners = torch.tensor(np.asarray(corners), dtype=torch.float32, device=self.device)
        self.wheel_points = torch.tensor(np.asarray(wheels), dtype=torch.float32, device=self.device)
        self.wheel_axes = torch.tensor(np.asarray(axes), dtype=torch.float32, device=self.device)
        self.geometry_bound = True

    def align_support(self, ids, root, joints, origins, surface_data, surface_valid, *, profile_ids=None):
        """Keep both wheels above terrain and one on its analytic support plane.

        Full nonwheel AABBs conservatively reject colliding angle/leg-bank
        combinations. This occurs before the caller's single state write.
        """
        ids = self._ids(ids)
        selected = self.selected[ids]
        if not bool(selected.any()):
            return
        if not self.geometry_bound:
            raise ValueError("Ordinary posture births require authenticated collision geometry")
        profiles = torch.zeros(len(ids), dtype=torch.long, device=self.device)
        if profile_ids is not None:
            profiles = profile_ids[ids] + 1
        lane = selected.nonzero().flatten()
        profiles = profiles[lane]
        if not torch.allclose(joints[lane], self.profile_positions[profiles], atol=2e-6, rtol=0.):
            raise ValueError("Ordinary posture cannot rotate an unbound reset joint configuration")
        local = root[lane, :2] - origins[ids[lane], :2]
        surfaces, valid = surface_data[ids[lane]], surface_valid[ids[lane]]
        quat = root[lane, 3:7]

        def required_height(quaternion):
            wheels = rotate_vectors(quaternion, self.wheel_points[profiles])
            axes = rotate_vectors(quaternion, self.wheel_axes[profiles])
            extents = self.wheel_radius[None, :, None] * (1. - axes.square()).clamp_min(0.).sqrt()
            extents += self.wheel_halfwidth[None, :, None] * axes.abs()
            wheel_heights = []
            for side in (0, 1):
                bounds = torch.stack((-extents[:, side, 0], extents[:, side, 0],
                    -extents[:, side, 1], extents[:, side, 1]), -1)
                support = support_height_for_footprint(surfaces, valid,
                    local + wheels[:, side, :2], bounds)
                wheel_heights.append(support - wheels[:, side, 2] + extents[:, side, 2])
            wheel_z = torch.stack(wheel_heights, -1).amax(-1)
            points = rotate_vectors(quaternion, self.collision_corners[profiles])
            lo, hi = points.amin(-2), points.amax(-2)
            body_z = torch.full_like(wheel_z, -torch.inf)
            for shape in range(lo.shape[1]):
                bounds = torch.stack((lo[:, shape, 0], hi[:, shape, 0],
                    lo[:, shape, 1], hi[:, shape, 1]), -1)
                support = support_height_for_footprint(surfaces, valid, local, bounds)
                body_z = torch.maximum(body_z, support - lo[:, shape, 2] + .001)
            return wheel_z, body_z

        wheel_z, body_z = required_height(quat)
        rejected = body_z > wheel_z + 1e-6
        if bool(rejected.any()):
            # A cancellation is explicit evidence, never a mid-episode repair.
            quat[rejected] = quat.new_tensor([0., 0., 0., 1.])
            root[lane[rejected], 3:7] = quat[rejected]
            self.angles[ids[lane[rejected]]] = 0.
            self.selected[ids[lane[rejected]]] = False
            self.birth_rejected_count += int(rejected.sum())
            wheel_z, body_z = required_height(quat)
            if bool((body_z > wheel_z + 1e-6).any()):
                raise ValueError("Ordinary posture leg-bank reset has no collision-free wheel support")
        root[lane, 2] = origins[ids[lane], 2] + wheel_z
        accepted = ~rejected
        self.birth_selected_count += int(accepted.sum())
        if bool(accepted.any()):
            self.birth_abs_angle_max_rad = max(self.birth_abs_angle_max_rad,
                float(self.angles[ids[lane[accepted]]].abs().amax()))

    def push_mask(self, due, ground, actor_owned, passive_hold, done):
        return due & self.eligible & ground & actor_owned & ~passive_hold & ~done

    def record_push(self, ids, before, after):
        ids = self._ids(ids)
        chosen = self.eligible[ids]
        if not bool(chosen.any()):
            return
        delta = after[chosen, :2] - before[chosen, :2]
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Ordinary posture native push readback is nonfinite")
        self.push_events += int(chosen.sum())
        self.push_velocity_delta_max_m_s = max(self.push_velocity_delta_max_m_s,
            float(delta.norm(dim=-1).amax()))

    def runtime_state(self):
        return runtime_expectation(self.settings, enabled=self.enabled)

    def observed(self, actor_update):
        relative, angle, speed = self.amplitudes(actor_update)
        return {"eligible_count": self.birth_population,
            "birth_eligible_population": self.birth_population,
            "relative_actor_updates": relative, "birth_angle_cap_rad": angle,
            "push_speed_cap_m_s": speed, "birth_selected_count": self.birth_selected_count,
            "birth_rejected_count": self.birth_rejected_count,
            "birth_abs_angle_max_rad": self.birth_abs_angle_max_rad,
            **self.timer_metrics,
            "push_events": self.push_events,
            "push_velocity_delta_max_m_s": self.push_velocity_delta_max_m_s,
            "geometry_bound": self.geometry_bound}

    def metrics(self, actor_update):
        return {"/ordinary_posture/" + key: float(value)
            for key, value in self.observed(actor_update).items()}
