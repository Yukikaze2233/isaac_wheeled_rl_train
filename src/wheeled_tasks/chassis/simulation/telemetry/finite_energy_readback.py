"""Observation-only rigid energy and explicit-work readback on native states.

Contact, constraint, passive damping and rotor exchange are not a complete work
balance here. Positive residuals identify evidence to inspect, not bad samples
or termination criteria. Resets and intentional velocity pushes are separate
classes, and a missing/aliased work estimate never hides a rigid energy jump.
The SDK net-contact field contains normal resultants; tangential friction is
separate and must not be omitted from a claimed total momentum balance.
"""
import math

import torch


COUNTERS = (
    "observed_lane_steps", "finite_energy_lane_steps", "valid_rigid_increment_lane_steps",
    "reset_lane_steps", "source_push_lane_steps", "valid_work_lane_steps",
    "angular_alias_lane_steps", "invalid_pressure_stamp_lane_steps", "unpaired_lane_steps",
)
PEAKS = ("rigid_energy_increment_j", "explicit_work_residual_j", "wheel_contact_force_n")


class FiniteEnergyReadback:
    """Cache startup native properties and only the previous physical sample."""

    def __init__(self, masses, inertias, *, physics_dt_s, angular_joints,
                 active_joint_ids, base_body_id, wheel_body_ids, pressure_body_ids=(),
                 configured_armature=None, velocity_push_column=None,
                 external_wrench_enabled=False, gravity_m_s2=9.81):
        if (masses.ndim != 2 or inertias.shape != (*masses.shape, 9)
                or masses.device != inertias.device or not math.isfinite(physics_dt_s)
                or physics_dt_s <= 0 or not math.isfinite(gravity_m_s2) or gravity_m_s2 < 0):
            raise ValueError("Energy readback requires ordered native masses/inertias and physical units")
        self.device = masses.device
        self.mass = masses.detach().double().clone()
        self.inertia = inertias.detach().double().reshape(*masses.shape, 3, 3).clone()
        symmetric = .5 * (self.inertia + self.inertia.transpose(-1, -2))
        a, b, c = symmetric[..., 0, :].unbind(-1)
        d, e, f = symmetric[..., 1, 1], symmetric[..., 1, 2], symmetric[..., 2, 2]
        # Sylvester's criterion avoids a CUDA eigensolver batch-size limit.
        positive = ((a > 0) & (a * d - b.square() > 0)
                    & (a * (d * f - e.square()) - b * (b * f - c * e)
                       + c * (b * e - c * d) > 0))
        if (not bool((torch.isfinite(self.mass) & (self.mass > 0)).all())
                or not bool(torch.isfinite(self.inertia).all())
                or not bool(positive.all())):
            raise ValueError("Energy readback requires finite positive native rigid-body properties")
        self.count, self.bodies = masses.shape
        self.angular = torch.as_tensor(angular_joints, dtype=torch.bool, device=self.device)
        self.active = torch.as_tensor(active_joint_ids, dtype=torch.long, device=self.device)
        self.wheels = torch.as_tensor(wheel_body_ids, dtype=torch.long, device=self.device)
        self.pressure = torch.as_tensor(pressure_body_ids, dtype=torch.long, device=self.device)
        if (self.angular.ndim != 1 or not self.angular.numel() or self.active.ndim != 1
                or not self.active.numel() or bool(((self.active < 0) | (self.active >= len(self.angular))).any())
                or type(base_body_id) is not int or not 0 <= base_body_id < self.bodies
                or self.wheels.ndim != 1 or not self.wheels.numel()
                or bool(((self.wheels < 0) | (self.wheels >= self.bodies)).any())
                or self.pressure.ndim != 1 or self.pressure.unique().numel() != self.pressure.numel()
                or bool(((self.pressure < 0) | (self.pressure >= self.bodies)).any())):
            raise ValueError("Energy readback requires actual native joint/body indices")
        self.armature = None
        if configured_armature is not None:
            self.armature = torch.as_tensor(configured_armature, dtype=torch.float64, device=self.device).clone()
            if (self.armature.shape != self.active.shape
                    or not bool((torch.isfinite(self.armature) & (self.armature >= 0)).all())):
                raise ValueError("Configured armature readback requires active-order nonnegative values")
        if velocity_push_column is not None and (type(velocity_push_column) is not int or velocity_push_column < 0):
            raise ValueError("Energy push provenance requires its explicit event column")
        self.base, self.dt, self.gravity = base_body_id, physics_dt_s, gravity_m_s2
        self.push_column, self.external = velocity_push_column, external_wrench_enabled
        self.previous, self.last_tick = None, None
        self.counts = torch.zeros(len(COUNTERS), dtype=torch.long, device=self.device)
        self.peaks = torch.full((len(PEAKS),), -torch.inf, dtype=torch.float64, device=self.device)
        self.classified_delta_peaks = torch.full((2,), -torch.inf, dtype=torch.float64, device=self.device)

    @staticmethod
    def _rotate(quaternion, value):
        q = quaternion / quaternion.norm(dim=-1, keepdim=True)
        twice = 2. * torch.linalg.cross(q[..., :3], value)
        return value + q[..., 3:] * twice + torch.linalg.cross(q[..., :3], twice)

    def _sample(self, state):
        keys = ("q", "dq", "episode_id", "body_link_pose_world", "body_com_pose_world",
                "body_com_linear_velocity_world", "body_com_angular_velocity_world")
        sample = {key: state[key].detach().clone() for key in keys}
        if (sample["q"].shape != (self.count, len(self.angular))
                or sample["dq"].shape != sample["q"].shape
                or sample["episode_id"].shape != (self.count,)
                or any(sample[key].shape != (self.count, self.bodies, width) for key, width in (
                    ("body_link_pose_world", 7), ("body_com_pose_world", 7),
                    ("body_com_linear_velocity_world", 3), ("body_com_angular_velocity_world", 3)))):
            raise ValueError("Energy state does not match cached native property order")
        position = sample["body_com_pose_world"][..., :3].double()
        velocity = sample["body_com_linear_velocity_world"].double()
        omega = sample["body_com_angular_velocity_world"].double()
        inverse = sample["body_link_pose_world"][..., 3:].double().clone()
        inverse[..., :3] *= -1
        local_omega = self._rotate(inverse, omega)
        body_linear = .5 * self.mass * velocity.square().sum(-1)
        body_angular = .5 * torch.einsum("nbi,nbij,nbj->nb", local_omega, self.inertia, local_omega)
        linear, angular = body_linear.sum(-1), body_angular.sum(-1)
        potential = (self.mass * self.gravity * position[..., 2]).sum(-1)
        sample.update(linear_kinetic_j=linear, angular_kinetic_j=angular, gravity_potential_j=potential,
                      rigid_energy_j=linear + angular + potential,
                      wheel_kinetic_energy_j=(body_linear + body_angular)[:, self.wheels])
        if self.armature is not None:
            sample["configured_rotor_kinetic_j"] = .5 * (
                self.armature * sample["dq"][:, self.active].double().square()).sum(-1)
        if self.push_column is not None:
            events = state["source_interval_event_counts"]
            if events.ndim != 2 or events.shape[0] != self.count or events.shape[1] <= self.push_column:
                raise ValueError("Energy state is missing the bound source-push counter")
            sample["push_count"] = events[:, self.push_column].detach().clone()
        return sample

    def observe(self, physical_tick, state, control_before):
        if type(physical_tick) is not int or self.last_tick is not None and physical_tick <= self.last_tick:
            raise ValueError("Energy readback requires increasing actual physical ticks")
        current = self._sample(state)
        previous = self.previous
        finite = torch.isfinite(current["rigid_energy_j"])
        zero = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        metrics = {key: current[key] for key in (
            "linear_kinetic_j", "angular_kinetic_j", "gravity_potential_j", "rigid_energy_j",
            "wheel_kinetic_energy_j")}
        if self.armature is not None:
            metrics["configured_rotor_kinetic_j"] = current["configured_rotor_kinetic_j"]
        paired = self.last_tick is not None and physical_tick == self.last_tick + 1
        reset, push, valid, work_valid, alias, bad_stamp = (zero.clone() for _ in range(6))
        nan = torch.full((self.count,), torch.nan, dtype=torch.float64, device=self.device)
        delta, actuator, pressure_work, external_work, residual = (nan.clone() for _ in range(5))
        if previous is not None:
            reset = current["episode_id"] != previous["episode_id"]
            if self.push_column is not None:
                push = current["push_count"] != previous["push_count"]
            delta = current["rigid_energy_j"] - previous["rigid_energy_j"]
            metrics["wheel_kinetic_energy_increment_j"] = (
                current["wheel_kinetic_energy_j"] - previous["wheel_kinetic_energy_j"])
            valid = finite & torch.isfinite(previous["rigid_energy_j"]) & ~reset & ~push & paired
            if self.armature is not None:
                metrics["configured_rotor_energy_change_j"] = (
                    current["configured_rotor_kinetic_j"] - previous["configured_rotor_kinetic_j"])
            if control_before is not None:
                dq = current["q"].double() - previous["q"].double()
                expected = .5 * self.dt * (current["dq"].double() + previous["dq"].double())
                windings = torch.round((dq - expected) / (2 * math.pi))
                dq = torch.where(self.angular, dq - 2 * math.pi * windings, dq)
                alias = ((self.dt * torch.maximum(current["dq"].abs(), previous["dq"].abs()) >= math.pi)
                         & self.angular).any(-1)
                effort = state["native_applied_effort"].double()
                actuator = (effort * dq).sum(-1)
                dx = (current["body_com_pose_world"][..., :3].double()
                      - previous["body_com_pose_world"][..., :3].double())
                omega = .5 * (current["body_com_angular_velocity_world"].double()
                               + previous["body_com_angular_velocity_world"].double())
                pressure_work = torch.zeros_like(actuator)
                if self.pressure.numel():
                    stamp = control_before["pressure_generated_physical_tick"].reshape(self.count, -1)[:, 0]
                    bad_stamp = stamp != physical_tick - 1
                    pressure_work = (control_before["pressure_forces_world"].double() * dx[:, self.pressure]).sum((1, 2))
                    pressure_work += self.dt * (control_before["pressure_torques_about_com_world"].double()
                                                * omega[:, self.pressure]).sum((1, 2))
                external_work = torch.zeros_like(actuator)
                if self.external:
                    quaternion = previous["body_link_pose_world"][:, self.base, 3:].double()
                    force = self._rotate(quaternion, control_before["source_held_force_body_n"].double())
                    torque = self._rotate(quaternion, control_before["source_held_torque_body_nm"].double())
                    external_work = (force * dx[:, self.base]).sum(-1)
                    external_work += self.dt * (torque * omega[:, self.base]).sum(-1)
                residual = delta - actuator - pressure_work - external_work
                work_valid = (valid & ~alias & ~bad_stamp & torch.isfinite(residual)
                    & torch.isfinite(current["dq"]).all(-1) & torch.isfinite(previous["dq"]).all(-1))
                metrics["angular_coordinate_wrap_corrections"] = ((windings != 0) & self.angular).sum(-1)
        metrics.update(rigid_energy_increment_j=delta, native_actuator_work_estimate_j=actuator,
            gas_pressure_work_estimate_j=pressure_work, held_external_work_estimate_j=external_work,
            explicit_work_residual_j=residual, valid_rigid_increment=valid, valid_work_estimate=work_valid,
            reset_between_samples=reset, source_velocity_push_between_samples=push,
            angular_displacement_aliased=alias, invalid_pressure_command_stamp=bad_stamp)
        wheel_force = state["all_partner_net_contact_force"][:, self.wheels].double().norm(dim=-1)
        wheel_peak, wheel_slot = wheel_force.max(-1)
        metrics.update(wheel_contact_force_n=wheel_peak, wheel_peak_body_id=self.wheels[wheel_slot],
                       wheel_contact_force_per_wheel_n=wheel_force)
        selected = torch.stack((torch.where(valid, delta, -torch.inf),
            torch.where(work_valid, residual, -torch.inf),
            torch.where(torch.isfinite(wheel_peak), wheel_peak, -torch.inf)))
        values, lanes = selected.max(-1)
        new_peaks = values > self.peaks
        self.peaks.copy_(torch.maximum(self.peaks, values))
        classified = torch.stack((torch.where(reset & torch.isfinite(delta), delta, -torch.inf),
                                  torch.where(push & ~reset & torch.isfinite(delta), delta, -torch.inf))).max(-1).values
        self.classified_delta_peaks.copy_(torch.maximum(self.classified_delta_peaks, classified))
        unpaired = torch.full_like(zero, not paired)
        self.counts += torch.stack(tuple(value.sum() for value in (
            torch.ones_like(zero), finite, valid, reset, push, work_valid, alias, bad_stamp, unpaired)))
        self.previous, self.last_tick = current, physical_tick
        return dict(metrics=metrics, previous=previous, current=current,
                    previous_tick=None if not paired else physical_tick - 1,
                    new_peaks=new_peaks, peak_values=values, peak_lane_ids=lanes)

    def summary(self):
        values = self.peaks.detach().cpu().tolist()
        classes = self.classified_delta_peaks.detach().cpu().tolist()
        return dict(enabled=True, scope="current native worker; all cached bodies at physical endpoints; COM energy and link-frame inertia",
            physics_dt_s=self.dt, native_properties="startup get_masses/get_inertias; immutable across episode resets",
            peak_values={key:value if math.isfinite(value) else None for key,value in zip(PEAKS, values)},
            counters=dict(zip(COUNTERS, self.counts.detach().cpu().tolist())),
            reset_energy_delta_peak_j=classes[0] if math.isfinite(classes[0]) else None,
            intentional_push_energy_delta_peak_j=classes[1] if math.isfinite(classes[1]) else None,
            classified_delta_scope="endpoint changes, not impulse work; overlapping reset/push is classified as reset",
            configured_armature_is_native_readback=False, configured_armature_available=self.armature is not None,
            wheel_force_scope="net normal contact resultant; separate tangential friction is not included",
            work_scope="estimates exclude full contact/constraint/passive work; positive residual is not a fault criterion",
            first_physics_endpoint="initializes energy cache; no comparison to possibly stale pre-step SDK velocities",
            numerical_or_sample_threshold_added=False, termination_changed=False, ppo_mask_changed=False)

    def metrics(self):
        finite = torch.isfinite(self.peaks)
        values = torch.where(finite, self.peaks, 0.)
        return {**{f"PhysicsEnergy/worker_peak_{key}": value.detach() for key,value in zip(PEAKS, values)},
                **{f"PhysicsEnergy/worker_peak_{key}_valid": value.detach() for key,value in zip(PEAKS, finite)},
                **{f"PhysicsEnergy/{key}": value.detach() for key,value in zip(COUNTERS, self.counts)}}
