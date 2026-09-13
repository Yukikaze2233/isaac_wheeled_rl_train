"""V40 nested-URDF path binding for the official Lab 3 PhysX contact sensor."""
from pxr import PhysxSchema, UsdPhysics

from isaaclab.sensors.contact_sensor import BaseContactSensor
from isaaclab_physx.physics import PhysxManager
from isaaclab_physx.sensors.contact_sensor import ContactSensor


class V40ContactSensor(ContactSensor):
    """Keep official buffers/kernels; resolve exact paths instead of flat leaf globs."""

    def __init__(self, cfg, *, stage, env_prim_paths, body_names):
        if cfg.filter_prim_paths_expr or cfg.track_contact_points or cfg.track_friction_forces:
            raise ValueError("V40 expects unfiltered rigid-body net contact forces")
        self._v40_body_paths = []
        self._v40_body_names = list(body_names)
        self._v40_env_count = len(env_prim_paths)
        prims = list(stage.Traverse())
        for env_path in env_prim_paths:
            bodies = {}
            for prim in prims:
                if (str(prim.GetPath()).startswith(env_path + "/Robot/")
                        and prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                    name = prim.GetName()
                    if name not in body_names or name in bodies:
                        raise ValueError(f"Unexpected or duplicate V40 contact body: {prim.GetPath()}")
                    bodies[name] = prim
            if set(bodies) != set(body_names):
                raise ValueError(f"Missing V40 contact bodies in {env_path}: {set(body_names) - set(bodies)}")
            for name in body_names:
                prim = bodies[name]
                # Lab's generic spawner stops at the first nested rigid body.
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
                self._v40_body_paths.append(str(prim.GetPath()))
        super().__init__(cfg)

    def _initialize_impl(self):
        BaseContactSensor._initialize_impl(self)
        if self._num_envs != self._v40_env_count:
            raise RuntimeError("V40 contact environment count mismatch")
        self._physics_sim_view = PhysxManager.get_physics_sim_view()
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(self._v40_body_paths)
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            self._v40_body_paths, filter_patterns=[],
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(self._v40_body_paths),
        )
        if list(self._body_physx_view.prim_paths) != self._v40_body_paths:
            raise RuntimeError("PhysX contact view changed the requested environment/body ordering")
        self._num_sensors = self._body_physx_view.count // self._num_envs
        if self._num_sensors != len(self._v40_body_names):
            raise RuntimeError("V40 contact view must expose all seven bodies in every clone")
        self._create_buffers()
