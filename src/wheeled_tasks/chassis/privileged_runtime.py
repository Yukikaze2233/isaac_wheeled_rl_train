"""PhysX readback and physical-step state for the opt-in 78D critic.

Native interfaces were checked against installed IsaacLab-3.0.0-beta2:
* isaaclab/envs/mdp/events.py:_RandomizeRigidBodyMaterialPhysx uses per-link
  rigid views' max_shapes, checks their sum against root_view.max_shapes, and
  passes a FULL CPU material buffer plus int32 environment ids to the setter.
* isaaclab_physx/assets/articulation/articulation_data.py:body_link_lin_vel_w
  is link-origin velocity, unlike get_link_velocities/body_com_vel_w.
* isaaclab/utils/math.py:quat_apply_inverse uses xyzw quaternions in this SDK.

No Isaac/Kit import or simulator startup occurs here. Material operations are
startup/reset operations and may synchronize; contact/velocity paths remain
Torch tensor operations. Existing native 81D behavior is never selected here.
"""

from dataclasses import dataclass, replace
import math

import torch


WHEEL_BODY_NAMES = ("L_link3", "R_link3")
WHEEL_MATERIAL_PRIOR_VERSIONS = ("wheel_shape_buckets_v1", "v7_low_grip_wheel_buckets_v2")


@dataclass(frozen=True)
class ContactMaterialPolicy:
    """Authoring policy, separate from sampled native material coefficients.

    flat baseline uses unit ground and multiply for friction and restitution. The
    wheel-bucket adapter represents this explicitly; it cannot reuse a legacy
    effective-ground coefficient or the average-combine inverse transform.
    Unselected legacy scenes retain their existing material authoring paths.
    """

    source_wheel_buckets: bool
    friction_combine: str
    restitution_combine: str
    robot_friction: float

    @classmethod
    def from_config(cls, config, *, legacy_contact_domain_active):
        selected = config.get("wheel_material_priors", {}).get("enabled", False)
        if type(selected) is not bool:
            raise ValueError("Wheel material selection must be a boolean")
        if selected:
            version = config["wheel_material_priors"].get("version", WHEEL_MATERIAL_PRIOR_VERSIONS[0])
            if version not in WHEEL_MATERIAL_PRIOR_VERSIONS:
                raise ValueError("Unknown wheel material prior version")
            if legacy_contact_domain_active:
                raise ValueError("Wheel material DR cannot stack with legacy ground friction DR")
            if config.get("friction_combine_mode") != "multiply":
                raise ValueError("Wheel material priors require explicit multiply friction combination")
            if config.get("restitution_combine_mode", "multiply") != "multiply":
                raise ValueError("Source wheel materials require multiply restitution combination")
            return cls(True, "multiply", "multiply", 1.)
        return cls(False, "multiply" if legacy_contact_domain_active else "average", "average",
                   1. if legacy_contact_domain_active else .5)

    def ground_friction(self, requested, *, triangle_mesh=False):
        if type(requested) not in (int, float) or not math.isfinite(requested) or requested < 0:
            raise ValueError("Ground friction must be finite and nonnegative")
        if self.source_wheel_buckets:
            if requested != 1.:
                raise ValueError("Source wheel buckets require unit ground; effective ground DR is a separate adapter")
            return 1.
        # Preserve legacy mesh authoring and contact-domain coefficients. Only
        # legacy boxes invert the average-combine effective coefficient.
        if triangle_mesh or self.friction_combine == "multiply":
            return requested
        return 2. * requested - .5

    def nominal_surfaces(self, surfaces, *, kind):
        """Make the source flat ground explicit in geometry metadata as well.

        terrain_surfaces and long-corridor Surface constructors use the legacy
        .5 default. Source wheel DR selects a different unit-ground owner. It
        cannot silently override a material course or a non-flat terrain.
        """
        if not self.source_wheel_buckets:
            return surfaces
        if (kind != "flat" or not surfaces
                or any(type(s.friction) not in (int, float)
                       or not math.isfinite(s.friction) or s.friction not in (.5, 1.)
                       for s in surfaces)):
            raise ValueError("Source unit-ground adapter requires nominal flat surfaces; material terrain needs review")
        return [replace(surface, friction=1.) for surface in surfaces]

    def summary(self):
        return {"scope": "authored_material_policy_pending_native_contact_qualification",
                "source_wheel_buckets": self.source_wheel_buckets,
                "friction_combine_mode": self.friction_combine,
                "restitution_combine_mode": self.restitution_combine,
                "robot_initial_friction": self.robot_friction,
                "ground": "unit_coefficient" if self.source_wheel_buckets else "legacy_surface_policy",
                "source_ground_adapter": "legacy_flat_default_to_explicit_unit" if self.source_wheel_buckets else None}


def verify_authored_combine_modes(stage, material_paths, policy):
    """Read the actual USD attributes, without claiming solver contact proof."""
    from pxr import PhysxSchema
    paths = tuple(material_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Material paths must be nonempty and unique")
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
            raise RuntimeError(f"Missing authored PhysX material: {path}")
        api = PhysxSchema.PhysxMaterialAPI(prim)
        for attr, expected in ((api.GetFrictionCombineModeAttr(), policy.friction_combine),
                               (api.GetRestitutionCombineModeAttr(), policy.restitution_combine)):
            if not attr.HasAuthoredValueOpinion() or attr.Get() != expected:
                raise RuntimeError(f"Material combine readback mismatch: {path}/{attr.GetName()}")
    return {"verified_material_count": len(paths), "evidence": "USD_authored_combine_attributes",
            "native_effective_contact_qualified": False}


def _as_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    import warp as wp
    return wp.to_torch(value)


def _cpu_env_ids(env_ids, count):
    if env_ids is None:
        return torch.arange(count, dtype=torch.long)
    ids = torch.as_tensor(env_ids)
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Environment ids must be a one-dimensional integer tensor")
    ids = ids.to(device="cpu", dtype=torch.long)
    if bool(((ids < 0) | (ids >= count)).any()) or len(ids.unique()) != len(ids):
        raise ValueError("Environment ids must be in range and unique")
    return ids


@dataclass(frozen=True)
class WheelShapeLayout:
    body_ids: tuple[int, int]
    shape_ranges: tuple[tuple[int, int], tuple[int, int]]
    shapes_per_body: tuple[int, ...]
    total_shapes: int


def wheel_shape_layout(robot, physics_sim_view) -> WheelShapeLayout:
    """Freeze named wheel ranges using the installed PhysX event-term pattern.

    The view must contain homogeneous clones. Do not infer material slots from
    body ids: bodies can have zero, one, or several collision shapes.
    """
    view = robot.root_view
    names = tuple(robot.body_names)
    paths = view.link_paths
    if not view.is_homogeneous or len(paths) != view.count or not paths:
        raise ValueError("Wheel material mapping requires a homogeneous articulation view")
    if len(set(names)) != len(names):
        raise ValueError("Articulation body names must be unique")
    for row in paths:
        if tuple(path.rsplit("/", 1)[-1] for path in row) != names:
            raise ValueError("Native link paths must match body_names in every clone")
    counts = []
    for path in paths[0]:
        link_view = physics_sim_view.create_rigid_body_view(path)
        if link_view.count != 1 or list(link_view.prim_paths) != [path]:
            raise ValueError(f"Rigid body view did not select exactly {path}")
        shapes = int(link_view.max_shapes)
        if shapes < 0:
            raise ValueError("Native shape counts must be nonnegative")
        counts.append(shapes)
    if sum(counts) != view.max_shapes:
        raise ValueError(f"Per-body shapes sum to {sum(counts)}, root max_shapes={view.max_shapes}")
    try:
        body_ids = tuple(names.index(name) for name in WHEEL_BODY_NAMES)
    except ValueError as exc:
        raise ValueError(f"Both named wheels {WHEEL_BODY_NAMES} are required") from exc
    ranges = tuple((sum(counts[:i]), sum(counts[:i + 1])) for i in body_ids)
    if any(start == end for start, end in ranges):
        raise ValueError("Each wheel must have at least one collision shape")
    return WheelShapeLayout(body_ids, ranges, tuple(counts), sum(counts))


class WheelMaterialCache:
    """Native [left/right, static/dynamic/restitution] startup/reset snapshot.

    Call refresh(ids) AFTER any external material setter. Multiple shapes on
    the same wheel must have identical material values; averaging or selecting
    an arbitrary shape would misrepresent the six privileged material slots.
    Never retain the native reusable host buffer as the critic cache.
    """

    def __init__(self, robot, physics_sim_view, device):
        self.root_view = robot.root_view
        self.count = self.root_view.count
        self.device = torch.device(device)
        self.layout = wheel_shape_layout(robot, physics_sim_view)
        self.wheel_body_ids = self.layout.body_ids
        self.values = torch.empty((self.count, 2, 3), dtype=torch.float32, device=self.device)
        self.refresh()

    def native_materials(self):
        """Return a detached full CPU copy suitable for the native sparse-row setter."""
        native = _as_torch(self.root_view.get_material_properties())
        expected = (self.count, self.layout.total_shapes, 3)
        if native.shape != expected or native.dtype != torch.float32:
            raise ValueError(f"Native materials must be float32 {expected}, got {native.shape}/{native.dtype}")
        return native.to(device="cpu").clone()

    def refresh(self, env_ids=None):
        ids = _cpu_env_ids(env_ids, self.count)
        native = self.native_materials()
        selected = []
        for name, (start, end) in zip(WHEEL_BODY_NAMES, self.layout.shape_ranges):
            wheel = native[ids, start:end]
            if not bool(torch.isfinite(wheel).all()):
                raise ValueError(f"{name} material contains non-finite values")
            if not torch.equal(wheel, wheel[:, :1].expand_as(wheel)):
                raise ValueError(f"{name} collision shapes have heterogeneous materials")
            selected.append(wheel[:, 0])
        values = torch.stack(selected, dim=1).to(device=self.device)
        # Update only after every wheel passed: a failing read cannot half-update.
        self.values[ids.to(self.device)] = values
        return self.values


def apply_startup_wheel_material_buckets(cache: WheelMaterialCache, *, seed: int,
                                         legacy_ground_randomization_enabled: bool,
                                         version: str = WHEEL_MATERIAL_PRIOR_VERSIONS[0]):
    """Versioned material sampling, with one of 64 buckets per wheel.

    Ground DR must already be disabled by the caller; this helper never changes
    ground or combine modes. All shapes on a wheel share its sample so the 78D
    six-slot material representation remains truthful. Buckets are sampled once
    by a private CPU generator; actual native readback must match the write.
    Do not call on every reset to continually create new PhysX material buckets.
    V2 widens the friction domain and anchors bucket0 at its lower endpoint;
    wheel assignments remain random, so a small batch may not use that bucket.
    """
    if version not in WHEEL_MATERIAL_PRIOR_VERSIONS:
        raise ValueError("Unknown wheel material prior version")
    if legacy_ground_randomization_enabled:
        raise ValueError("Wheel material DR cannot stack with legacy ground friction DR")
    if getattr(cache, "_startup_buckets_applied", False):
        raise RuntimeError("Startup material buckets were already applied to this cache")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    low_grip = version == WHEEL_MATERIAL_PRIOR_VERSIONS[1]
    low = torch.tensor([.2, .2, .02] if low_grip else [.5, .4, .02])
    high = torch.tensor([1.2, 1., .2])
    buckets = low + (high - low) * torch.rand((64, 3), generator=generator)
    buckets[:, 1] = torch.minimum(buckets[:, 0], buckets[:, 1])
    if low_grip:
        buckets[0, :2] = .2
    bucket_ids = torch.randint(64, (cache.count, 2), generator=generator)
    desired = buckets[bucket_ids]
    materials = cache.native_materials()
    for i, (start, end) in enumerate(cache.layout.shape_ranges):
        materials[:, start:end] = desired[:, i:i + 1]
    # Vendor setter requires the full material table, even for sparse env ids.
    import warp as wp
    ids = torch.arange(cache.count, dtype=torch.int32)
    cache.root_view.set_material_properties(
        wp.from_torch(materials.contiguous(), dtype=wp.float32),
        wp.from_torch(ids, dtype=wp.int32),
    )
    cache.refresh()
    if not torch.equal(cache.values.cpu(), desired):
        raise RuntimeError("Native wheel material readback differs from requested buckets")
    cache._startup_buckets_applied = True
    actual = cache.values.cpu().reshape(-1, 3)
    return {"version": version, "buckets": buckets, "bucket_ids": bucket_ids,
            "sampled_static_friction_range": [float(actual[:, 0].min()), float(actual[:, 0].max())],
            "sampled_dynamic_friction_range": [float(actual[:, 1].min()), float(actual[:, 1].max())],
            "low_friction_wheel_count": int(((actual[:, 0] < .5) | (actual[:, 1] < .4)).sum()),
            "low_friction_definition": "static<0.5 or dynamic<0.4, below the previous domain",
            "lower_endpoint_wheel_count": int(((actual[:, 0] == .2) & (actual[:, 1] == .2)).sum())}


class WheelContactHistory:
    """15ms wheel net-force history: training 3x5ms, deployment replay 15x1ms."""

    def __init__(self, count, device, physics_dt_s=.005):
        if physics_dt_s not in (.005, .001):
            raise ValueError("78D contact history supports only 200Hz training or 1kHz replay")
        self.count = count
        self.history = torch.zeros((round(.015 / physics_dt_s), count, 2), dtype=torch.float32, device=device)
        self._cursor = 0

    def observe(self, force_norm_N2):
        """Call exactly once per physics step, after native contact readback.

        Input is norm(sum of contact force vectors) per wheel, NOT the sum of
        individual contact-force norms and NOT just the vertical component.
        Runtime finite monitoring remains in the environment's batched audit.
        """
        if force_norm_N2.shape != (self.count, 2):
            raise ValueError("Wheel net-force norms must be (environments, 2)")
        if force_norm_N2.device != self.history.device:
            raise ValueError("Contact history and force norms must share a device")
        if not force_norm_N2.is_floating_point():
            raise TypeError("Wheel force norms must be floating point")
        self.history[self._cursor].copy_(force_norm_N2)
        self._cursor = (self._cursor + 1) % len(self.history)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.history.zero_()
        else:
            self.history[:, env_ids] = 0

    def contacts(self):
        return self.history.amax(dim=0) > 5.


def wheel_relative_velocity_body(robot_data, wheel_ids, root_link_linear_velocity_w):
    """R(root)^-1 (v_wheel_link_origin - v_root_link_origin), left then right.

    wheel_ids should be WheelMaterialCache.wheel_body_ids. Use the explicit
    SDK link-origin accessor, never native get_link_velocities (COM velocities).
    This follows the upstream relative-vector definition without subtracting
    omega x r. It retains y; the 78D builder owns the documented y=0 projection.
    """
    if len(wheel_ids) != 2:
        raise ValueError("Exactly two ordered wheel body ids are required")
    velocity = _as_torch(robot_data.body_link_lin_vel_w)[:, wheel_ids]
    quaternion = _as_torch(robot_data.root_link_quat_w)
    count = root_link_linear_velocity_w.shape[0]
    if velocity.shape != (count, 2, 3) or root_link_linear_velocity_w.shape != (count, 3):
        raise ValueError("Wheel/root link-origin velocities must be (N,2,3)/(N,3)")
    if quaternion.shape != (count, 4):
        raise ValueError("Root link orientation must be (N,4) xyzw")
    relative = velocity - root_link_linear_velocity_w[:, None]
    xyz = quaternion[:, None, :3].expand(-1, 2, -1)
    t = 2. * torch.cross(xyz, relative, dim=-1)
    return relative - quaternion[:, None, 3:4] * t + torch.cross(xyz, t, dim=-1)
