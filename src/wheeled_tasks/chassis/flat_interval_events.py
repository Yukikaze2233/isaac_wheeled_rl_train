"""flat baseline interval disturbances adapted to the installed SDK6 interfaces.

Origin: reference b8ff79f3 0719 env.yaml, custom events.apply_external_force_torque_xyz,
and IsaacLab EventManager/DirectRLEnv. Interval clocks span episode resets in
that DirectRLEnv call chain. Push is a world COM velocity INCREMENT, despite
the upstream function name. A sampled body-local wrench persists until its
next event or an articulation reset. No Kit import occurs in this module.

The archive does not bind its original IsaacLab commit. This implementation
names the installed-SDK adaptation; it is not a bitwise historical replay.
"""
from dataclasses import dataclass
import math

import torch


VERSION = "scut0719-interval-sdk6-cpu-v1"
TERM_ORDER = ("push_robot", "base_external_force_torque_xyz")


def validate_interval_selection(config):
    if "flat_interval_events" not in config:
        return None
    settings = config["flat_interval_events"]
    if not isinstance(settings, dict) or settings != {"version": VERSION}:
        raise ValueError("Unknown source interval component contract")
    if "flat_episode_runtime" not in config:
        raise ValueError("Source interval events require exclusive source flat episode ownership")
    return dict(settings)


@dataclass(frozen=True)
class IntervalEventBatch:
    push_ids: torch.Tensor
    push_velocity_world: torch.Tensor
    push_increment_world: torch.Tensor
    wrench_ids: torch.Tensor
    force_body_n: torch.Tensor
    torque_body_nm: torch.Tensor


class FlatIntervalEvents:
    """Two per-lane continuous interval clocks, in archive declaration order.

    Float32 subtraction and the <1e-6 trigger match EventManager. No ground,
    command mode, terminal or zero-torque gate is added. Call after reset and
    before returned observations; the paid transition must already be saved.
    """

    def __init__(self, count, device, generator):
        if type(count) is not int or count <= 0:
            raise ValueError("Interval events need a positive environment count")
        self.count, self.device = count, torch.device(device)
        self._validate_generator(generator)
        self.time_left_s = torch.stack([self._interval(count, generator) for _ in TERM_ORDER])
        self.force_body_n = torch.zeros(count, 1, 3, device=self.device)
        self.torque_body_nm = torch.zeros_like(self.force_body_n)
        self.event_count = torch.zeros(2, count, dtype=torch.long, device=self.device)
        self.policy_ticks = 0

    def _validate_generator(self, generator):
        if not isinstance(generator, torch.Generator) or generator.device != self.device:
            raise ValueError("Interval events and explicit RNG must share a device")

    def _interval(self, count, generator):
        return torch.rand(count, device=self.device, generator=generator) * 5. + 5.

    def _ids(self, ids):
        value = torch.as_tensor(ids, device=self.device)
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Event reset ids must be a one-dimensional integer tensor")
        if len(value.unique()) != len(value) or bool(((value < 0) | (value >= self.count)).any()):
            raise ValueError("Event reset ids must be unique and in range")
        return value.long()

    def reset(self, ids):
        """Articulation reset clears the held wrench, not interval countdowns."""
        ids = self._ids(ids)
        self.force_body_n[ids] = 0.
        self.torque_body_nm[ids] = 0.

    def advance(self, dt, root_com_velocity_world, generator):
        if type(dt) not in (int, float) or not math.isclose(dt, .02, rel_tol=0., abs_tol=1e-12):
            raise ValueError("Source interval events advance once per 20ms policy tick")
        self._validate_generator(generator)
        velocity = root_com_velocity_world
        if (not isinstance(velocity, torch.Tensor) or velocity.shape != (self.count, 6)
                or velocity.device != self.device or velocity.dtype != torch.float32
                or not bool(torch.isfinite(velocity).all())):
            raise ValueError("Source push requires finite float32 world COM velocities (N,6)")
        self.time_left_s[0] -= dt
        push_ids = (self.time_left_s[0] < 1e-6).nonzero(as_tuple=False).flatten()
        self.time_left_s[0, push_ids] = self._interval(len(push_ids), generator)
        # Upstream draws all six columns, including four zero-range columns.
        # Drawing just XY would change the source term's RNG consumption.
        low = velocity.new_tensor([-.25, -.25, 0., 0., 0., 0.])
        high = velocity.new_tensor([.25, .25, 0., 0., 0., 0.])
        delta = low + (high - low) * torch.rand((len(push_ids), 6),
            device=self.device, generator=generator)
        pushed = velocity[push_ids].clone() + delta
        self.event_count[0, push_ids] += 1

        self.time_left_s[1] -= dt
        wrench_ids = (self.time_left_s[1] < 1e-6).nonzero(as_tuple=False).flatten()
        self.time_left_s[1, wrench_ids] = self._interval(len(wrench_ids), generator)
        # The custom source event samples fx, fy, fz, tx, ty, tz sequentially.
        forces = torch.cat([torch.rand((len(wrench_ids), 1, 1), device=self.device,
            generator=generator) * 20. - 10. for _ in range(3)], dim=-1)
        torques = torch.cat([torch.rand((len(wrench_ids), 1, 1), device=self.device,
            generator=generator) * 2. - 1. for _ in range(3)], dim=-1)
        self.force_body_n[wrench_ids] = forces
        self.torque_body_nm[wrench_ids] = torques
        self.event_count[1, wrench_ids] += 1
        self.policy_ticks += 1
        return IntervalEventBatch(push_ids.clone(), pushed, delta, wrench_ids.clone(), forces, torques)

    def diagnostics(self):
        return {"interval_time_left_s": self.time_left_s.clone(), "event_count": self.event_count.clone(),
                "held_force_body_n": self.force_body_n.clone(), "held_torque_body_nm": self.torque_body_nm.clone(),
                "policy_ticks": self.policy_ticks}

    def apply_to_robot(self, robot, base_body_id, generator, *, root_kinematics=None):
        """Apply a post-reset batch via the installed SDK6 public setters.

        Native root velocity is read and verified. Wrench receipt is the public
        composer body-frame buffer, not a solver force measurement. No physics
        tick, sensor age, command origin or SDK simulation timestamp advances.
        A failed write/readback aborts the caller; it cannot retry as a new tick.
        """
        from .privileged_runtime import _as_torch
        if (type(base_body_id) is not int or not 0 <= base_body_id < len(robot.body_names)
                or robot.body_names[base_body_id] != "base_link"):
            raise ValueError("Source external wrench requires the actual named base_link")
        velocity = _as_torch(robot.root_view.get_root_velocities()).clone()
        batch = self.advance(.02, velocity, generator)
        if len(batch.push_ids):
            robot.write_root_com_velocity_to_sim_index(root_velocity=batch.push_velocity_world,
                env_ids=batch.push_ids, full_data=False)
            readback = _as_torch(robot.root_view.get_root_velocities())[batch.push_ids]
            if not torch.equal(readback, batch.push_velocity_world):
                raise RuntimeError("Source push native COM velocity readback mismatch")
            if root_kinematics is not None:
                root_kinematics.invalidate_velocity_after_native_write()
        if len(batch.wrench_ids):
            composer = robot.permanent_wrench_composer
            body_ids = torch.tensor([base_body_id], dtype=torch.long, device=self.device)
            composer.set_forces_and_torques_index(forces=batch.force_body_n, torques=batch.torque_body_nm,
                env_ids=batch.wrench_ids, body_ids=body_ids, is_global=False)
            composer.compose_to_body_frame()
            actual_force = _as_torch(composer.out_force_b)[batch.wrench_ids, base_body_id:base_body_id + 1]
            actual_torque = _as_torch(composer.out_torque_b)[batch.wrench_ids, base_body_id:base_body_id + 1]
            if not torch.equal(actual_force, batch.force_body_n) or not torch.equal(actual_torque, batch.torque_body_nm):
                raise RuntimeError("Source wrench composer buffer readback mismatch")
        return batch

    def metrics(self):
        return {"/domain/source_push_count": self.event_count[0].sum(),
                "/domain/source_wrench_count": self.event_count[1].sum(),
                "/domain/source_held_force_abs_max_n": self.force_body_n.abs().max(),
                "/domain/source_held_torque_abs_max_nm": self.torque_body_nm.abs().max()}

    def state_dict(self):
        return {"version": VERSION, "count": self.count, **self.diagnostics()}

    def load_state_dict(self, state):
        expected = set(self.state_dict())
        if (not isinstance(state, dict) or set(state) != expected or state["version"] != VERSION
                or type(state["count"]) is not int or state["count"] != self.count
                or type(state["policy_ticks"]) is not int
                or state["policy_ticks"] < 0):
            raise ValueError("Invalid source interval checkpoint")
        fields = (("interval_time_left_s", self.time_left_s), ("event_count", self.event_count),
                  ("held_force_body_n", self.force_body_n), ("held_torque_body_nm", self.torque_body_nm))
        for name, target in fields:
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.dtype != target.dtype or value.device != target.device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"Invalid source interval checkpoint tensor: {name}")
        if (bool((state["event_count"] < 0).any()) or bool((state["interval_time_left_s"] > 10.).any())
                or bool((state["interval_time_left_s"] <= 0).any())
                or bool((state["held_force_body_n"].abs() > 10.).any())
                or bool((state["held_torque_body_nm"].abs() > 1.).any())):
            raise ValueError("Source interval checkpoint is outside its parameter domain")
        for name, target in fields:
            target.copy_(state[name])
        self.policy_ticks = state["policy_ticks"]
