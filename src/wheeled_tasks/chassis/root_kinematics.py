"""Root link velocity with startup-static native COM and explicit read epochs.

Model writes must call invalidate_model() before changing masses/inertias/COM,
then refresh_model() only after the final native setter. Recreated views need a
new adapter. Dynamic epochs advance only at the existing robot.update() calls:
same-epoch root velocity/pose writes deliberately retain already-read caches.
"""
from __future__ import annotations

import torch
import warp as wp


class StaticRootLinkKinematics:
    def __init__(self, robot, count, device, *, kernels=None):
        if kernels is None:
            from isaaclab_physx.assets import kernels
        self.robot, self.count, self.device = robot, count, device
        self.kernels = kernels
        self.read_epoch = 0
        self.world_epoch = self.body_epoch = -1
        self.model_ready = False
        self._link_velocity = wp.zeros(count, dtype=wp.spatial_vectorf, device=device)
        self._link_tensor = wp.to_torch(self._link_velocity)
        # Preserve the vendor's spatial-vector view, including its stride.
        self._world_velocity = wp.from_torch(self._link_tensor[:, :3], dtype=wp.vec3f)
        self._body_velocity = wp.zeros(count, dtype=wp.vec3f, device=device)
        self._body_tensor = wp.to_torch(self._body_velocity)
        self.refresh_model()

    def invalidate_model(self):
        self.model_ready = False
        self.world_epoch = self.body_epoch = -1

    def refresh_model(self):
        """Snapshot native root COM after all mechanical writes have completed."""
        native = wp.to_torch(self.robot.root_view.get_coms())
        if native.ndim != 3 or native.shape[0] != self.count or native.shape[-1] != 7 or native.shape[1] < 1:
            raise ValueError("Native COM layout must be [environments, bodies, xyz+xyzw]")
        if not bool(torch.isfinite(native[:, 0]).all()):
            raise ValueError("Native root COM contains nonfinite values")
        # The vendor root kernel uses body index 0. Clone the final native value;
        # never retain its reused host buffer or pre-inertia principal axes.
        self.local_com = native[:, :1].to(device=self.device).contiguous().clone()
        self._local_com = wp.from_torch(self.local_com, dtype=wp.transformf)
        self.model_ready = True
        self.world_epoch = self.body_epoch = -1

    def advance_read_epoch(self):
        self.read_epoch += 1

    def update_robot_data(self, dt):
        """Public hook for native updates and observations after epoch advance."""
        self.robot.update(dt)
        if dt > 0:
            self.advance_read_epoch()

    def linear_velocity_world(self):
        if not self.model_ready:
            raise RuntimeError("Mechanical writes require a final native COM refresh")
        if self.world_epoch != self.read_epoch:
            wp.launch(self.kernels.get_root_link_vel_from_root_com_vel,
                dim=self.count,
                inputs=[self.robot.data.root_com_vel_w.warp,
                        self.robot.data.root_link_pose_w.warp, self._local_com],
                outputs=[self._link_velocity], device=self.device)
            self.world_epoch = self.read_epoch
        return self._link_tensor[:, :3]

    def linear_velocity_body(self):
        if not self.model_ready:
            raise RuntimeError("Mechanical writes require a final native COM refresh")
        if self.body_epoch != self.read_epoch:
            self.linear_velocity_world()
            wp.launch(self.kernels.quat_apply_inverse_1D_kernel,
                dim=self.count,
                inputs=[self._world_velocity, self.robot.data.root_link_quat_w.warp],
                outputs=[self._body_velocity], device=self.device)
            self.body_epoch = self.read_epoch
        return self._body_tensor


def root_link_linear_velocity(env, *, frame="world"):
    if frame not in ("world", "body"):
        raise ValueError("Root link velocity frame must be world or body")
    cache = getattr(env, "root_kinematics", None)
    if cache is None:
        name = "root_link_lin_vel_w" if frame == "world" else "root_link_lin_vel_b"
        return getattr(env.robot.data, name).torch
    return cache.linear_velocity_world() if frame == "world" else cache.linear_velocity_body()


def update_robot_data(env, dt):
    """Advance the adapter only when the native data update has succeeded."""
    cache = getattr(env, "root_kinematics", None)
    if cache is None:
        env.robot.update(dt)
    else:
        cache.update_robot_data(dt)


def robot_data_update_hook(env):
    """Return the update hook whose observers run after the read epoch advances."""
    cache = getattr(env, "root_kinematics", None)
    return (env.robot, "update") if cache is None else (cache, "update_robot_data")
